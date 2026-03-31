# engine/xi_selector.py
# Constrained optimiser: squad of 16-18 -> best Playing 11 satisfying PSL rules.
# Weather modifiers applied to player scores before optimisation.
# Returns 3 alternative XIs (A = primary, B = spin-heavy, C = pace-heavy).
#
# PSL constraints:
#   - Exactly 11 players
#   - Max 4 overseas players
#   - Min 1 wicketkeeper
#   - Min 4 genuine bowlers (to cover 20 overs with part-timers)
#   - Min 5 batters (positions 1-5)
#
# Public API:
#   select_xi(squad, venue, weather, innings, model_path) -> list[XiOption]

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.situation import WeatherImpact

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

PROJ_ROOT    = Path(__file__).resolve().parent.parent
MODEL_PATH   = PROJ_ROOT / "models" / "saved" / "xi_scorer.pkl"
PLAYER_INDEX = PROJ_ROOT / "data" / "processed" / "player_index_2026_enriched.csv"
PLAYER_INDEX_FALLBACK = PROJ_ROOT.parent / "player_index_2026_enriched.csv"
STATS_PATH        = PROJ_ROOT / "data" / "processed" / "player_stats.parquet"
PARTNERSHIP_PATH  = PROJ_ROOT / "data" / "processed" / "partnership_history.parquet"
RECENT_FORM_PATH  = PROJ_ROOT / "data" / "processed" / "recent_form.parquet"

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

XI_SIZE          = 11
MAX_OVERSEAS     = 4
MIN_KEEPERS      = 1
MIN_BOWLERS      = 4    # genuine (role = Bowler or All-rounder with bowling)
MIN_BATTERS      = 5    # pure batters + keeper
MIN_SPINNERS     = 1    # PSL pitches deteriorate; a spinner is almost always needed
MIN_EMERGING     = 1    # PSL rule: exactly 1 U23 uncapped Pakistani emerging player
MAX_EMERGING     = 1

# Fix 4.4: Venues where pace dominates — spinner minimum relaxed to 0.
# Rawalpindi and Peshawar produce the most seam movement in the PSL.
# At these venues the "must include a spinner" constraint can be waived
# when there is no spinner in the squad worth picking ahead of a pacer.
_PACE_DOMINANT_VENUES = {
    "Rawalpindi Cricket Stadium",
    "Arbab Niaz Stadium, Peshawar",
    "Arbab Niaz Stadium",
}

# Module-level cache so TabNet/XGBoost model is loaded only once per process.
_CACHED_MODEL_PAYLOAD: dict | None = None
_CACHED_MODEL_KEY: str | None = None


def _min_spinners_for_venue(venue: str) -> int:
    """Return minimum spinners required at this venue (1 everywhere; 0 at seam tracks)."""
    return 0 if venue in _PACE_DOMINANT_VENUES else MIN_SPINNERS

ROLE_KEEPER   = {"Wicketkeeper", "WK-Batsman"}
ROLE_BOWLER   = {"Bowler"}
ROLE_ALLROUND = {"All-rounder"}
ROLE_BATTER   = {"Batsman"}


# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class ScoredPlayer:
    player_name:      str
    score:            float
    role:             str
    is_overseas:      bool
    is_keeper:        bool
    is_bowler:        bool
    is_allrounder:    bool
    batting_style:    str
    bowling_style:    str
    form_coefficient: float = 1.0   # 0.80–1.20 based on recent PSL season trends
    form_tag:         str   = ""    # "In form" | "Out of form" | "Good form" | ""
    model_source:     str   = ""    # "tabnet" | "transfer" | "analytical" | "standard"
    is_emerging:      bool  = False # PSL rule: U23 uncapped Pakistani player


@dataclass
class XiPlayer:
    batting_position: int
    player_name:      str
    role:             str
    score:            float
    key_stat:         str     # one-line highlight for the brief
    model_source:     str = ""  # Upgrade 5: "tabnet"|"transfer"|"analytical"|"standard"


@dataclass
class XiOption:
    label:          str         # "Option A" | "Option B" | "Option C"
    description:    str         # brief label: "Primary XI" | "Spin-heavy" | "Pace-heavy"
    players:        list[XiPlayer]
    overseas_count: int
    bowler_count:   int
    constraint_note:str         # e.g. "4 overseas, 5 bowlers, 22 overs coverage"
    total_score:    float


# ---------------------------------------------------------------------------
# PLAYER META LOADER
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_meta(path: str) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    if not Path(path).exists():
        return meta
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("player_name", "").strip()
            if not name:
                continue
            style = (row.get("bowling_style") or "").lower()
            def _f(key: str, default: float = 0.0) -> float:
                v = row.get(key)
                try:
                    return float(v) if v and str(v).strip() not in ("", "nan") else default
                except (ValueError, TypeError):
                    return default
            meta[name] = {
                "primary_role":       row.get("primary_role",  "Batsman").strip(),
                "batting_style":      row.get("batting_style", "Right-hand bat").strip(),
                "bowling_style":      row.get("bowling_style", "").strip(),
                "is_overseas":        row.get("is_overseas", "False").strip().lower() == "true",
                "is_emerging":        row.get("is_emerging", "False").strip().lower() == "true",
                "is_pace":  any(w in style for w in ("fast","medium","seam","swing","pace")),
                "is_spin":  any(w in style for w in ("spin","off","leg","googly","chinaman","slow")),
                # T20 career stats — proxy for players with no/limited PSL history
                "t20_career_sr":      _f("t20_career_sr",      120.0),
                "t20_career_avg":     _f("t20_career_avg",       15.0),
                "t20_career_economy": _f("t20_career_economy",    8.5),
                "data_tier":          int(_f("data_tier", 3)),   # 3 = T20 estimate (safe unknown default)
                # IPL stats — preferred over generic T20 career for overseas players
                # (subcontinent conditions closely match PSL pitches/weather)
                "ipl_matches":        int(_f("ipl_matches",        0.0)),
                "ipl_career_sr":      _f("ipl_career_sr",          0.0),
                "ipl_career_avg":     _f("ipl_career_avg",         0.0),
                "ipl_career_economy": _f("ipl_career_economy",     0.0),
                # New columns from player_index_2026_enriched.csv
                "national_t20_economy": _f("national_t20_economy", 0.0),
                "national_t20_sr":      _f("national_t20_sr",      0.0),
                "bat_sr_set":           _f("bat_sr_set",            0.0),
                "bat_sr_chase":         _f("bat_sr_chase",          0.0),
                "innings_sr_delta":     _f("innings_sr_delta",      0.0),
                "bowl_dot_pct":         _f("bowl_dot_pct",          0.0),
            }
    return meta


# ---------------------------------------------------------------------------
# PROXY BUILDER — IPL preferred over generic T20 for overseas players
# ---------------------------------------------------------------------------

def _build_proxy(m: dict, innings: int = 1) -> dict:
    """
    Build the t20_proxy dict passed to score_player().
    Prefers IPL stats over generic T20 career stats when:
      - Batting : player has >= 10 IPL matches  (reliable sample)
      - Bowling : player has >=  5 IPL matches  (sufficient for economy)
    IPL conditions (subcontinent pitches, heat/humidity) closely match PSL,
    making IPL a better predictor than career T20 averages blended across
    Australia, England, Caribbean etc.
    Confidence tier is bumped to 2 (70%) for any player with >= 5 IPL matches.

    innings-specific SR: bat_sr_chase used when chasing (innings==2),
    bat_sr_set when setting (innings==1). Falls back to IPL then
    t20_career_sr when innings-split data is absent (value == 0.0).
    """
    ipl_n     = m.get("ipl_matches", 0)
    ipl_sr    = m.get("ipl_career_sr",      0.0)
    ipl_avg   = m.get("ipl_career_avg",     0.0)
    ipl_eco   = m.get("ipl_career_economy", 0.0)

    use_ipl_bat  = ipl_n >= 10 and ipl_sr  > 0
    use_ipl_bowl = ipl_n >= 5  and ipl_eco > 0
    has_ipl      = ipl_n >= 5

    # Use innings-specific SR when available and sample is sufficient.
    # bat_sr_set / bat_sr_chase are 0.0 when data is absent — treat as missing.
    if innings == 2 and m.get("bat_sr_chase", 0.0) > 0:
        chosen_bat_sr = m["bat_sr_chase"]
    elif innings == 1 and m.get("bat_sr_set", 0.0) > 0:
        chosen_bat_sr = m["bat_sr_set"]
    else:
        chosen_bat_sr = None   # fall through to existing IPL / t20_career logic

    return {
        "bat_sr":        ipl_sr if use_ipl_bat else (
            chosen_bat_sr if chosen_bat_sr else m.get("t20_career_sr", 120.0)
        ),
        "bat_avg":       ipl_avg if use_ipl_bat  else m.get("t20_career_avg",       15.0),
        "bowl_economy":  ipl_eco if use_ipl_bowl else m.get("t20_career_economy",    8.5),
        "data_tier":     2       if has_ipl       else m.get("data_tier",              1),
        "bowling_style": m.get("bowling_style", ""),
    }


# ---------------------------------------------------------------------------
# FORM COEFFICIENT
# ---------------------------------------------------------------------------

def _load_form_coefficients(
    players:    list[str],
    meta:       dict[str, dict],
    stats_path: Path | None = None,
) -> dict[str, tuple[float, str]]:
    """
    Compute a form coefficient [0.80, 1.20] for each player based on their
    recent PSL season performance vs career baseline.

    Weighting of recent seasons:
        2025 → 50%   (last PSL season — most predictive)
        2024 → 30%
        2023 → 20%

    Batter form (for players with bat_innings >= 3 in a recent season):
        avg_ratio    = recent_bat_avg    / career_bat_avg
        sr_ratio     = recent_bat_sr     / career_bat_sr
        bndry_ratio  = recent_boundary_pct / career_boundary_pct
        season_form  = avg_ratio * 0.50 + sr_ratio * 0.35 + bndry_ratio * 0.15
        (Matches xi_scorer model weights: avg 30% / sr 35% / boundary_pct 15%,
        rescaled to 3-feature sum since pp_sr / death_sr are not season-level stats.)

    Known limitation: form coefficient uses career/season averages without
    innings-context split (chasing avg vs setting avg). Some batters perform
    significantly better in chases (e.g. Babar Azam) vs setting totals.
    This split requires innings-level tagging in player_stats.parquet.
    TODO: add chase_bat_avg / set_bat_avg columns to player_stats schema.

    Bowler form (for players with bowl_overs >= 5 in a recent season):
        eco_ratio = career_bowl_economy / recent_bowl_economy   (inverted — lower eco = better)
        season_form = eco_ratio

    All-rounders: blend of batter and bowler form (50/50).
    If fewer than 2 seasons of data exist, returns 1.0 (no adjustment).
    Final coefficient clamped to [0.80, 1.20].

    Tags (recalibrated to ±30% range):
        form ≥ 1.20  → "In form"      (clear standout recent season)
        form ≥ 1.08  → "Good form"
        form ≤ 0.80  → "Out of form"  (significant drop)
        form ≤ 0.92  → "Below par"
        else          → ""             (normal variance)
    Final coefficient clamped to [0.70, 1.30].
    """
    path = stats_path or STATS_PATH
    result: dict[str, tuple[float, str]] = {p: (1.0, "") for p in players}

    try:
        df = pd.read_parquet(path)
        overall = df[df["phase"] == "overall"]
    except Exception:
        return result

    def _season_row(player: str, season: int):
        rows = overall[(overall["player_name"] == player) & (overall["season"] == season)]
        return rows.iloc[0] if not rows.empty else None

    def _safe(row, col: str, default: float = 0.0) -> float:
        if row is None:
            return default
        v = row.get(col)
        try:
            return float(v) if pd.notna(v) else default
        except (ValueError, TypeError):
            return default

    SEASON_WEIGHTS = [(2025, 0.50), (2024, 0.30), (2023, 0.20)]

    for player in players:
        m    = meta.get(player, {})
        role = m.get("primary_role", "Batsman")
        is_bowler     = role in ("Bowler",)
        is_allrounder = role in ("All-rounder",)
        is_batter     = not is_bowler

        career = _season_row(player, 0)
        if career is None:
            continue  # no PSL data at all — leave at 1.0

        career_avg = _safe(career, "bat_avg",      0.0)
        career_sr  = _safe(career, "bat_sr",       0.0)
        career_eco = _safe(career, "bowl_economy", 0.0)

        weighted_bat   = 0.0
        weighted_bowl  = 0.0
        total_bat_w    = 0.0
        total_bowl_w   = 0.0

        for season, weight in SEASON_WEIGHTS:
            row = _season_row(player, season)
            if row is None:
                continue

            # Batting form — avg 50%, sr 35%, boundary_pct 15% (matches xi_scorer model)
            innings = _safe(row, "bat_innings", 0.0)
            career_boundary_pct = _safe(career, "boundary_pct", 0.0)
            if innings >= 3 and career_avg > 0 and career_sr > 0:
                s_avg          = _safe(row, "bat_avg",       career_avg)
                s_sr           = _safe(row, "bat_sr",        career_sr)
                s_boundary_pct = _safe(row, "boundary_pct",  career_boundary_pct)
                avg_ratio      = min(1.5, s_avg / career_avg)
                sr_ratio       = min(1.5, s_sr  / career_sr)
                # boundary_pct ratio: fall back gracefully if career value missing
                if career_boundary_pct > 0:
                    bndry_ratio = min(1.5, s_boundary_pct / career_boundary_pct)
                else:
                    bndry_ratio = 1.0
                bat_form  = avg_ratio * 0.50 + sr_ratio * 0.35 + bndry_ratio * 0.15
                weighted_bat  += bat_form * weight
                total_bat_w   += weight

            # Bowling form — economy AND wickets per over (career vs recent)
            overs = _safe(row, "bowl_overs", 0.0)
            if overs >= 5:
                s_eco  = _safe(row, "bowl_economy", career_eco)
                s_wkts = _safe(row, "bowl_wickets", 0.0)

                # Economy ratio (lower eco = better, so inverted)
                eco_ratio = min(1.5, career_eco / s_eco) if (s_eco > 0 and career_eco > 0) else 1.0

                # Wickets-per-over ratio
                career_wpo  = _safe(career, "bowl_wickets", 0.0) / max(1.0, _safe(career, "bowl_overs", 1.0))
                season_wpo  = s_wkts / max(1.0, overs)
                wpo_ratio   = min(1.5, season_wpo / career_wpo) if career_wpo > 0 else 1.0

                # Blend: 60% economy + 40% wickets
                season_bowl = eco_ratio * 0.60 + wpo_ratio * 0.40
                weighted_bowl  += season_bowl * weight
                total_bowl_w   += weight

        bat_coeff  = (weighted_bat  / total_bat_w)  if total_bat_w  > 0 else 1.0
        bowl_coeff = (weighted_bowl / total_bowl_w) if total_bowl_w > 0 else 1.0

        if is_bowler:
            # Pure bowler: bowling form drives the coefficient
            # Batting form still included at 20% if data exists (tail batters matter)
            if total_bat_w > 0:
                coeff = bowl_coeff * 0.80 + bat_coeff * 0.20
            else:
                coeff = bowl_coeff
        elif is_allrounder:
            # All-rounder: equal weight when both available
            if total_bat_w > 0 and total_bowl_w > 0:
                coeff = bat_coeff * 0.50 + bowl_coeff * 0.50
            elif total_bat_w > 0:
                coeff = bat_coeff
            elif total_bowl_w > 0:
                coeff = bowl_coeff
            else:
                coeff = 1.0
        else:
            # Pure batter / WK-batsman
            coeff = bat_coeff if total_bat_w > 0 else 1.0

        # Clamp to ±30% — allows genuine form surges (83% SR improvement → full 30% credit)
        # Previous ±20% was suppressing real signal on players with standout recent seasons.
        coeff = round(max(0.70, min(1.30, coeff)), 4)

        # Human-readable tag — recalibrated to match the wider range
        if coeff >= 1.20:
            tag = "In form"        # clear recent standout
        elif coeff >= 1.08:
            tag = "Good form"      # above-par recent season
        elif coeff <= 0.80:
            tag = "Out of form"    # significant recent drop
        elif coeff <= 0.92:
            tag = "Below par"      # below recent baseline
        else:
            tag = ""               # within normal variance

        result[player] = (coeff, tag)

    return result


# ---------------------------------------------------------------------------
# RECENT FORM LOADER
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load_recent_form_df(path: str) -> "pd.DataFrame | None":
    """Load recent_form.parquet and return it, or None if unavailable."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def _recent_form_lookup(
    player: str,
    role: str,
    venue: str,
    rf_path: str,
) -> tuple[float, str, float]:
    """
    Return (form_score, form_trend, venue_form_score) for a player.

    form_score:       0-100 based on bat/bowl/blend recent form.
    form_trend:       "rising" | "stable" | "declining"
    venue_form_score: 0-100 for the match venue (50 = neutral / insufficient data).
    """
    df = _load_recent_form_df(rf_path)
    if df is None:
        return 50.0, "stable", 50.0

    # Overall row (venue == "")
    overall = df[(df["player_name"] == player) & (df["venue"] == "")]
    if overall.empty:
        return 50.0, "stable", 50.0

    row   = overall.iloc[0]
    is_bowler     = role in ("Bowler",)
    is_allrounder = role in ("All-rounder",)

    if is_bowler:
        form_score = float(row.get("bowl_form_score", 50.0))
        form_trend = str(row.get("bowl_trend", "stable"))
    elif is_allrounder:
        form_score = (float(row.get("bat_form_score", 50.0)) +
                      float(row.get("bowl_form_score", 50.0))) / 2.0
        bat_t  = str(row.get("bat_trend", "stable"))
        bowl_t = str(row.get("bowl_trend", "stable"))
        if bat_t == "rising" and bowl_t == "rising":
            form_trend = "rising"
        elif bat_t == "declining" and bowl_t == "declining":
            form_trend = "declining"
        else:
            form_trend = "stable"
    else:
        form_score = float(row.get("bat_form_score", 50.0))
        form_trend = str(row.get("bat_trend", "stable"))

    # Venue-specific row
    venue_form = 50.0
    if venue:
        vrow = df[(df["player_name"] == player) & (df["venue"] == venue)]
        if not vrow.empty:
            vm = int(vrow.iloc[0].get("venue_matches", 0))
            if vm >= 3:
                venue_form = float(vrow.iloc[0].get("venue_form_score", 50.0))

    return round(form_score, 2), form_trend, round(venue_form, 2)


# ---------------------------------------------------------------------------
# INNINGS CONTEXT (CHASE / SET) SCORING
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load_innings_context(stats_path: str) -> dict[str, dict]:
    """
    Cache {player_name: {bat_avg_chase, bat_avg_set, bat_sr_chase, bat_sr_set,
                          bat_innings_chase, bat_innings_set, innings_context_split}}
    from the career (season=0) overall row in player_stats.parquet.
    Returns {} if the parquet is missing or the columns haven't been built yet.
    """
    try:
        df = pd.read_parquet(stats_path)
    except Exception:
        return {}

    career_overall = df[
        (df["season"] == 0) & (df["phase"] == "overall")
    ].copy()

    required = {"bat_avg_chase", "bat_avg_set", "innings_context_split"}
    if not required.issubset(career_overall.columns):
        return {}   # pipeline not yet rebuilt with the new columns

    result: dict[str, dict] = {}
    for _, row in career_overall.iterrows():
        name = row.get("player_name")
        if not name:
            continue
        def _flt(val, default=0.0):
            try:
                v = float(val)
                return default if pd.isna(v) else v
            except (TypeError, ValueError):
                return default

        result[name] = {
            "bat_avg_chase":        _flt(row.get("bat_avg_chase")),
            "bat_avg_set":          _flt(row.get("bat_avg_set")),
            "bat_sr_chase":         _flt(row.get("bat_sr_chase")),
            "bat_sr_set":           _flt(row.get("bat_sr_set")),
            "bat_innings_chase":    int(_flt(row.get("bat_innings_chase"))),
            "bat_innings_set":      int(_flt(row.get("bat_innings_set"))),
            "innings_context_split":_flt(row.get("innings_context_split")),
        }
    return result


_MIN_CONTEXT_INNINGS = 5   # fewer innings → fall back to career bat_avg


def _innings_context_bonus(
    player:   str,
    innings:  int,
    ctx_map:  dict[str, dict],
) -> tuple[float, str]:
    """
    Return (score_adjustment, metadata_note) based on the player's chase/set split.

    Adjustment is applied for batters/allrounders only (bowlers: 0.0).
    Scale: each 5-run difference in avg → ±1 point, capped at ±5.
    Falls back to 0.0 when:
      - no context data loaded (pipeline not rebuilt yet)
      - fewer than _MIN_CONTEXT_INNINGS in the relevant context
    """
    data = ctx_map.get(player)
    if not data:
        return 0.0, "ctx:no_data"

    if innings == 2:     # chasing
        n      = data["bat_innings_chase"]
        split  = data["innings_context_split"]   # positive = better chaser
        label  = "ctx:chase"
    else:                # setting (innings == 1)
        n      = data["bat_innings_set"]
        split  = -data["innings_context_split"]  # invert so positive = better setter
        label  = "ctx:set"

    if n < _MIN_CONTEXT_INNINGS:
        return 0.0, f"ctx:fallback(<{_MIN_CONTEXT_INNINGS}inn)"

    adjustment = round(max(-5.0, min(5.0, split / 5.0)), 2)
    note = f"{label}:{adjustment:+.1f}"
    return adjustment, note


# ---------------------------------------------------------------------------
# PLAYER SCORING
# ---------------------------------------------------------------------------

def _matchup_bonus(
    player:              str,
    role:                str,
    meta:                dict[str, dict],
    opp_lh_pct:          float,
    opp_batters:         list[str],
    opp_spin_economies:  dict[str, float] | None = None,
) -> float:
    """
    Compute a matchup-context score bonus/penalty for bowlers.

    Signals:
      1. Left-arm vs LH-heavy lineup:  +5 (opp ≥50% LH), +3 (≥40% LH)
      2. Off-spin vs LH-heavy lineup:  +4 (≥50%), +2 (≥40%)  — turns into body
      3. Spin subtype vs opposition economy (per-style, NOT lumped):
           eco < 7.0  →  -8  (they score freely vs this spin type — avoid)
           eco > 9.5  →  +8  (they struggle vs this spin type — prioritise)
           eco < 7.5  →  -4  (below average ease vs this type)
           eco > 8.5  →  +4  (above average difficulty vs this type)

         Subtype mapping (uses opp_spin_economies keys):
           leg-spin / googly / chinaman → "legspin"
           left-arm orthodox / slow LA  → "leftarm_spin"
           off-spin / off-break / others → "offspin"

         Falls back to generic "spin" if subtype key absent (pre-2026 data).

    Only applied to Bowlers and All-rounders.
    """
    if role not in ("Bowler", "All-rounder"):
        return 0.0

    bowling_style = meta.get(player, {}).get("bowling_style", "").lower()
    if not bowling_style:
        return 0.0

    bonus = 0.0
    spin_eco = opp_spin_economies or {}

    is_left_arm = "left" in bowling_style
    is_off_spin = any(w in bowling_style for w in ("off spin", "off-spin", "off break"))
    is_spin_bowler = any(w in bowling_style for w in
                         ("spin", "off", "leg", "break", "orthodox", "chinaman", "googly", "slow"))

    # --- Signal 1 & 2: Left-hand lineup angle ---
    if is_left_arm and opp_lh_pct >= 50:
        bonus += 5.0
    elif is_left_arm and opp_lh_pct >= 40:
        bonus += 3.0

    if is_off_spin and opp_lh_pct >= 50:
        bonus += 4.0
    elif is_off_spin and opp_lh_pct >= 40:
        bonus += 2.0

    # --- Signal 3: Spin subtype economy matchup ---
    if is_spin_bowler and spin_eco:
        # Determine subtype key
        if any(w in bowling_style for w in ("leg", "googly", "chinaman")):
            eco = spin_eco.get("legspin", spin_eco.get("spin", 8.0))
        elif is_left_arm and any(w in bowling_style for w in ("orthodox", "spin", "slow")):
            eco = spin_eco.get("leftarm_spin", spin_eco.get("spin", 8.0))
        else:
            eco = spin_eco.get("offspin", spin_eco.get("spin", 8.0))

        if eco < 7.0:
            bonus -= 8.0   # opposition scores freely vs this spin style — avoid
        elif eco < 7.5:
            bonus -= 4.0   # below average difficulty for them
        elif eco > 9.5:
            bonus += 8.0   # they genuinely struggle vs this style — prioritise
        elif eco > 8.5:
            bonus += 4.0   # above average difficulty for them

    return bonus


def _score_squad(
    squad:              list[str],
    venue:              str,
    innings:            int,
    weather:            WeatherImpact,
    meta:               dict[str, dict],
    payload:            dict,
    stats_path:         Path | None = None,
    opp_lh_pct:         float = 33.0,
    opp_batters:        list[str] | None = None,
    opp_spin_economies: dict[str, float] | None = None,
) -> list[ScoredPlayer]:
    """
    Score every player in the squad for this match context.

    Layers applied on top of base XGBoost score:
      1. Recent form blend: final_score = career_score * 0.60 + form_score * 0.40
         (form_score 0-100 from recent_form.parquet; falls back to season-form
         coefficient if parquet is unavailable).
      2. Venue form modifier (+10%) when player has ≥3 matches at this venue
         and a strong venue_form_score.
      3. Matchup bonus (+0–8) for bowlers who exploit opposition LH/RH composition.
    """
    from models.train_xi_scorer import score_player

    form_map = _load_form_coefficients(squad, meta, stats_path)
    opp_bat  = opp_batters or []
    rf_path  = str(RECENT_FORM_PATH)

    # Chase / set context — load once per squad evaluation
    _ctx_stats_path = str(stats_path) if stats_path else str(STATS_PATH)
    _ctx_map        = _load_innings_context(_ctx_stats_path)

    # Phase-aware spinner penalty:
    # Dew builds up during the innings and is WORST when it arrives early (many overs affected).
    # Late dew (onset over 16+) only affects 4 overs — spinner value is mostly preserved.
    # Early dew (onset over 6) affects 14+ overs — heavy overall penalty warranted.
    #
    # dew_fraction = proportion of the innings AFFECTED by dew = (20 - onset_over) / 20
    #   onset=6  → fraction=0.70 → heavy blend toward raw penalty  (correct — most overs affected)
    #   onset=14 → fraction=0.30 → light blend toward raw penalty   (correct — few overs affected)
    #   onset=0  → no dew → fraction=0 → no adjustment
    #
    # eff_penalty = raw * dew_fraction + 1.0 * (1 - dew_fraction)
    # e.g. onset=6,  raw=0.40 → eff = 0.40×0.70 + 1.0×0.30 = 0.58  (heavier than late dew)
    # e.g. onset=14, raw=0.60 → eff = 0.60×0.30 + 1.0×0.70 = 0.88  (lenient — late arrival)
    raw_spin_pen  = weather.spinner_penalty
    onset_over    = weather.dew_onset_over
    if onset_over > 0 and raw_spin_pen < 1.0:
        # Fraction of innings exposed to dew (early onset = larger fraction = heavier penalty)
        dew_fraction      = max(0.0, min(1.0, (20 - onset_over) / 20.0))
        _eff_spin_penalty = raw_spin_pen * dew_fraction + 1.0 * (1.0 - dew_fraction)
        # Cap: effective penalty cannot exceed the raw WeatherImpact value
        effective_spin_penalty = max(raw_spin_pen, round(_eff_spin_penalty, 3))
    else:
        effective_spin_penalty = raw_spin_pen

    scored: list[ScoredPlayer] = []
    for p in squad:
        m       = meta.get(p, {})
        role    = m.get("primary_role", "Batsman")
        is_spin = m.get("is_spin", False)
        is_pace = m.get("is_pace", False)

        # Swing bonus: delegated to WeatherImpact.seam_swing_bonus() which gives the
        # full bonus to swing/seam specialists and 35% to pure pace bowlers.
        style_lower = m.get("bowling_style", "").lower()
        if is_pace:
            eff_swing_bonus = weather.seam_swing_bonus(style_lower)
        else:
            eff_swing_bonus = 1.0

        _src: list = []
        base_score = score_player(
            player          = p,
            venue           = venue,
            innings_num     = innings,
            payload         = payload,
            spinner_penalty = effective_spin_penalty if is_spin else 1.0,
            swing_bonus     = eff_swing_bonus,
            pace_bounce     = weather.pace_bounce_bonus if is_pace else 1.0,
            role_override   = role,
            t20_proxy       = _build_proxy(m, innings=innings),
            _source_out     = _src,
        )

        form_coeff, _legacy_tag = form_map.get(p, (1.0, ""))

        # Blend career score (60%) with recent form score (40%) from recent_form.parquet.
        # Falls back to the season-form coefficient if parquet unavailable.
        rf_score, rf_trend, rf_venue = _recent_form_lookup(p, role, venue, rf_path)
        if rf_score != 50.0 or rf_trend != "stable":
            # Recent form data available — use 60/40 blend
            blended = base_score * 0.60 + rf_score * 0.40
            # Venue form modifier: +10% when ≥3 matches at this venue with strong record
            if rf_venue >= 65.0:
                blended *= 1.10
        else:
            # Fallback: existing season-form coefficient
            blended = base_score * form_coeff

        # Build form tag from recent form score + trend
        if rf_score >= 70 and rf_trend == "rising":
            form_tag = "In form"
        elif rf_score >= 55:
            form_tag = "Good form"
        elif rf_score < 40 and rf_trend == "declining":
            form_tag = "Out of form"
        elif rf_score < 40:
            form_tag = "Below par"
        else:
            form_tag = _legacy_tag   # fall back to season-based tag when recent form is neutral

        matchup_b = _matchup_bonus(p, role, meta, opp_lh_pct, opp_bat, opp_spin_economies)

        # Innings context bonus: reward players who historically excel in this
        # specific innings context (chasing vs setting).
        # Only meaningful for batters/allrounders; bowlers get 0.0.
        _is_batting_role = role not in ("Bowler",)
        if _is_batting_role:
            ctx_adj, ctx_note = _innings_context_bonus(p, innings, _ctx_map)
        else:
            ctx_adj, ctx_note = 0.0, ""

        sc = round(blended + matchup_b + ctx_adj, 2)

        _base_src = _src[0] if _src else ""
        _model_src = f"{_base_src}|{ctx_note}" if ctx_note else _base_src

        scored.append(ScoredPlayer(
            player_name      = p,
            score            = sc,
            role             = role,
            is_overseas      = m.get("is_overseas",  False),
            is_keeper        = role in ROLE_KEEPER,
            is_bowler        = role in ROLE_BOWLER,
            is_allrounder    = role in ROLE_ALLROUND,
            batting_style    = m.get("batting_style", "Right-hand bat"),
            bowling_style    = m.get("bowling_style", ""),
            form_coefficient = form_coeff,
            form_tag         = form_tag,
            model_source     = _model_src,
            is_emerging      = m.get("is_emerging",  False),
        ))

    return sorted(scored, key=lambda x: x.score, reverse=True)


# ---------------------------------------------------------------------------
# CONSTRAINT VALIDATOR
# ---------------------------------------------------------------------------

_SPIN_KEYWORDS = ("spin", "off", "leg", "break", "orthodox", "chinaman", "googly", "slow")

def _is_spinner(p: ScoredPlayer) -> bool:
    style = p.bowling_style.lower()
    return any(w in style for w in _SPIN_KEYWORDS) and (p.is_bowler or p.is_allrounder)


def _validate(selected: list[ScoredPlayer], venue: str = "") -> tuple[bool, str]:
    """Check PSL constraints. Returns (is_valid, reason_if_invalid)."""
    overseas = sum(1 for p in selected if p.is_overseas)
    keepers  = sum(1 for p in selected if p.is_keeper)
    bowlers  = sum(1 for p in selected if p.is_bowler or p.is_allrounder)
    spinners = sum(1 for p in selected if _is_spinner(p))
    emerging = sum(1 for p in selected if p.is_emerging)

    if overseas > MAX_OVERSEAS:
        return False, f"Too many overseas: {overseas} > {MAX_OVERSEAS}"
    if keepers < MIN_KEEPERS:
        return False, f"No wicketkeeper in XI"
    if bowlers < MIN_BOWLERS:
        return False, f"Insufficient bowlers: {bowlers} < {MIN_BOWLERS}"
    # Spinner minimum is venue-dependent (0 at seam-dominant venues like Rawalpindi/Peshawar)
    min_spin = _min_spinners_for_venue(venue)
    if spinners < min_spin:
        return False, f"No spinner in XI — PSL pitches typically require at least one spinner"
    if emerging < MIN_EMERGING:
        return False, f"No emerging player in XI — PSL rule requires exactly 1 U23 uncapped Pakistani"
    if emerging > MAX_EMERGING:
        return False, f"Too many emerging players: {emerging} > {MAX_EMERGING}"
    return True, ""


# ---------------------------------------------------------------------------
# GREEDY OPTIMISER
# ---------------------------------------------------------------------------

def _greedy_select(
    scored:         list[ScoredPlayer],
    prefer_spin:    bool = False,
    prefer_pace:    bool = False,
    forced_players: Optional[list[str]] = None,
    venue:          str = "",
) -> list[ScoredPlayer]:
    """
    Greedy selection: pick highest-scoring players satisfying constraints.

    Strategy:
      0. Forced players (captain, must-plays) are locked in first
      1. Must-haves first: best keeper, enough domestic players to satisfy overseas cap
      2. Fill with highest-scoring players that don't violate constraints
      3. If bowlers are short, swap in the best available bowler/allrounder
    """
    selected:  list[ScoredPlayer] = []
    remaining: list[ScoredPlayer] = list(scored)

    overseas_used = 0
    keeper_added  = False
    bowlers_added = 0

    def _can_add(p: ScoredPlayer) -> bool:
        nonlocal overseas_used
        if p.is_overseas and overseas_used >= MAX_OVERSEAS:
            return False
        return True

    # Step 0: lock in forced players (captain / must-plays)
    forced_set = set(forced_players or [])
    if forced_set:
        forced_in_squad = [p for p in remaining if p.player_name in forced_set]
        for p in forced_in_squad:
            if len(selected) >= XI_SIZE:
                break
            selected.append(p)
            remaining.remove(p)
            if p.is_overseas:
                overseas_used += 1

    # Apply preference bias for Spin-heavy / Pace-heavy options.
    # Uses a large enough boost (+22) and penalty (-8) to force real swaps.
    # Spin detection covers all spin styles: off-break, leg-break, orthodox, chinaman, googly.
    # Pace detection is restricted to dedicated pace bowlers (fast/seam) — excludes
    # medium-pace all-rounders who are already counted as batting options.
    if prefer_spin or prefer_pace:
        SPIN_KEYWORDS = ("spin", "off", "leg", "break", "orthodox", "chinaman", "googly", "slow")
        PACE_KEYWORDS = ("fast", "seam", "swing")  # excludes "medium" to avoid boosting pace-AR batters

        def _is_spin_bowler(p: ScoredPlayer) -> bool:
            style = p.bowling_style.lower()
            return any(w in style for w in SPIN_KEYWORDS) and (p.is_bowler or p.is_allrounder)

        def _is_pace_bowler(p: ScoredPlayer) -> bool:
            style = p.bowling_style.lower()
            return any(w in style for w in PACE_KEYWORDS) and (p.is_bowler or p.is_allrounder)

        def _bias(p: ScoredPlayer) -> float:
            if prefer_spin:
                if _is_spin_bowler(p):
                    return p.score + 22.0   # strongly prefer spin bowlers/AR
                if _is_pace_bowler(p) and p.is_bowler:
                    return p.score - 8.0    # slight penalty for pure pace bowlers
            if prefer_pace:
                if _is_pace_bowler(p):
                    return p.score + 22.0   # strongly prefer pace bowlers/AR
                if _is_spin_bowler(p) and p.is_bowler:
                    return p.score - 8.0    # slight penalty for pure spin bowlers
            return p.score

        remaining = sorted(remaining, key=_bias, reverse=True)

    # Step 1: ensure keeper
    keepers = [p for p in remaining if p.is_keeper]
    if keepers:
        best_keeper = keepers[0]
        selected.append(best_keeper)
        remaining.remove(best_keeper)
        if best_keeper.is_overseas:
            overseas_used += 1
        keeper_added = True

    # Step 2: fill greedily
    emerging_used = sum(1 for p in selected if p.is_emerging)  # count forced emerging already
    for p in remaining[:]:
        if len(selected) >= XI_SIZE:
            break
        if not _can_add(p):
            continue
        # Cap emerging at MAX_EMERGING during greedy fill; Step 6 ensures exactly 1 is present
        if p.is_emerging and emerging_used >= MAX_EMERGING:
            continue
        selected.append(p)
        if p.is_overseas:
            overseas_used += 1
        if p.is_emerging:
            emerging_used += 1
        if p.is_bowler or p.is_allrounder:
            bowlers_added += 1
        remaining.remove(p)

    # Step 3: enforce minimum bowlers — swap out lowest-scoring non-bowler if needed
    current_bowlers = sum(1 for p in selected if p.is_bowler or p.is_allrounder)
    if current_bowlers < MIN_BOWLERS:
        need = MIN_BOWLERS - current_bowlers
        available_bowlers = sorted(
            [p for p in remaining if (p.is_bowler or p.is_allrounder) and _can_add(p)],
            key=lambda x: x.score,
            reverse=True,
        )
        # Remove lowest-scoring non-bowling batters to make room
        non_bowlers_in_xi = sorted(
            [p for p in selected if not p.is_bowler and not p.is_allrounder and not p.is_keeper],
            key=lambda x: x.score,
        )
        for i in range(min(need, len(available_bowlers), len(non_bowlers_in_xi))):
            selected.remove(non_bowlers_in_xi[i])
            selected.append(available_bowlers[i])
            if available_bowlers[i].is_overseas:
                overseas_used += 1

    # Step 4: enforce minimum spinner — venue-conditional (Fix 4.4)
    # At seam-dominant venues (Rawalpindi, Peshawar) the spinner minimum is 0.
    current_spinners = sum(1 for p in selected if _is_spinner(p))
    if current_spinners < _min_spinners_for_venue(venue):
        available_spinners = sorted(
            [p for p in remaining if _is_spinner(p) and _can_add(p)],
            key=lambda x: x.score,
            reverse=True,
        )
        if available_spinners:
            # Remove lowest-scoring pure batter (non-keeper, non-bowler) to make room
            swap_out_candidates = sorted(
                [p for p in selected if not p.is_bowler and not p.is_allrounder and not p.is_keeper],
                key=lambda x: x.score,
            )
            if swap_out_candidates:
                selected.remove(swap_out_candidates[0])
                selected.append(available_spinners[0])
                if available_spinners[0].is_overseas:
                    overseas_used += 1

    # Step 5: Enforce EXACTLY 4 overseas players.
    # Must run BEFORE emerging enforcement so the overseas swap cannot accidentally
    # remove the only emerging player (the emerging guard below then re-inserts if needed).
    MIN_OVERSEAS = 4
    current_overseas = [p for p in selected if p.is_overseas]
    if len(current_overseas) < MIN_OVERSEAS:
        need_overseas = MIN_OVERSEAS - len(current_overseas)
        available_overseas = sorted(
            [p for p in remaining if p.is_overseas],
            key=lambda x: x.score,
            reverse=True,
        )
        is_bowl = lambda x: x.is_bowler or x.is_allrounder
        for i in range(min(need_overseas, len(available_overseas))):
            locals_in_xi = sorted(
                [p for p in selected if not p.is_overseas], key=lambda x: x.score
            )
            for local_p in locals_in_xi:
                # Do not drop below keeper minimum
                if local_p.is_keeper and sum(1 for x in selected if x.is_keeper) <= MIN_KEEPERS:
                    if not available_overseas[i].is_keeper:
                        continue
                # Do not drop below bowler minimum
                if is_bowl(local_p) and sum(1 for x in selected if is_bowl(x)) <= MIN_BOWLERS:
                    if not is_bowl(available_overseas[i]):
                        continue
                # Safe to swap (emerging guard runs afterwards in Step 6)
                selected.remove(local_p)
                selected.append(available_overseas[i])
                # Update overseas_used so _can_add rejects further overseas in Step 6
                overseas_used += 1
                # Remove from remaining so Step 6 cannot pick the same player again
                if available_overseas[i] in remaining:
                    remaining.remove(available_overseas[i])
                break

    # Step 6: enforce exactly 1 emerging player (PSL rule: 1 U23 uncapped Pakistani).
    # Runs AFTER overseas enforcement so overseas count is already locked.
    # 6a. If none selected → force best-scoring emerging player from remaining.
    # 6b. If 2+ selected → keep only the best, replace extras with non-emerging players.
    current_emerging = [p for p in selected if p.is_emerging]
    emerging_count   = len(current_emerging)

    if emerging_count == 0:
        available_emerging = sorted(
            [p for p in remaining if p.is_emerging],
            key=lambda x: x.score,
            reverse=True,
        )
        if available_emerging:
            # Prefer swapping out lowest-scoring non-essential batter
            swap_out = sorted(
                [p for p in selected
                 if not p.is_keeper and not p.is_bowler and not p.is_allrounder
                 and not p.is_emerging and not p.is_overseas],
                key=lambda x: x.score,
            )
            if swap_out:
                selected.remove(swap_out[0])
                selected.append(available_emerging[0])
            else:
                # Last resort — swap lowest non-keeper, non-emerging (may be overseas)
                swap_out_any = sorted(
                    [p for p in selected if not p.is_keeper and not p.is_emerging],
                    key=lambda x: x.score,
                )
                if swap_out_any and len(selected) == XI_SIZE:
                    removed = swap_out_any[0]
                    selected.remove(removed)
                    if removed.is_overseas:
                        overseas_used -= 1
                    selected.append(available_emerging[0])
                elif len(selected) < XI_SIZE:
                    selected.append(available_emerging[0])

    elif emerging_count > MAX_EMERGING:
        # Keep only the best-scoring emerging player; replace extras
        current_emerging_sorted = sorted(current_emerging, key=lambda x: x.score, reverse=True)
        for excess in current_emerging_sorted[MAX_EMERGING:]:
            replacements = sorted(
                [p for p in remaining if not p.is_emerging and _can_add(p)],
                key=lambda x: x.score,
                reverse=True,
            )
            if replacements:
                selected.remove(excess)
                remaining.remove(replacements[0])
                selected.append(replacements[0])
                if replacements[0].is_overseas:
                    overseas_used += 1

    return selected[:XI_SIZE]


# ---------------------------------------------------------------------------
# PARTNERSHIP DATA
# ---------------------------------------------------------------------------

def _load_partnerships(path: Path | None = None) -> dict[tuple[str, str], dict]:
    """
    Load career partnership data from partnership_history.parquet.
    Returns a bidirectional lookup: {(playerA, playerB): {avg_sr, avg_runs, occurrences}}.
    Only career rows (season == 0) are loaded.
    """
    lookup: dict[tuple[str, str], dict] = {}
    p = path or PARTNERSHIP_PATH
    try:
        df  = pd.read_parquet(p)
        dfC = df[df["season"] == 0]
        for _, row in dfC.iterrows():
            a    = str(row.get("batter1", "")).strip()
            b    = str(row.get("batter2", "")).strip()
            if not a or not b:
                continue
            data = {
                "avg_sr":       float(row.get("avg_sr",      0.0) or 0.0),
                "avg_runs":     float(row.get("avg_runs",    0.0) or 0.0),
                "occurrences":  float(row.get("occurrences", 0.0) or 0.0),
                "avg_over_broken": float(row.get("avg_over_when_broken", 10.0) or 10.0),
            }
            # Store both directions so lookups work regardless of order
            lookup[(a, b)] = data
            lookup[(b, a)] = data
    except Exception:
        pass
    return lookup


def _partnership_score(a: str, b: str, lookup: dict) -> float:
    """
    Return a partnership quality score (0–100) for pair (a, b).
    Formula: occurrences × avg_sr / 15  (capped at 100).
    Requires at least 2 occurrences to return a non-zero score.
    """
    data = lookup.get((a, b)) or lookup.get((b, a))
    if data is None or data["occurrences"] < 2:
        return 0.0
    return min(100.0, data["occurrences"] * data["avg_sr"] / 15.0)


def _best_partner_note(player: str, others: list[str], lookup: dict) -> str:
    """
    Return a short note about the player's best known batting partner
    from the list of other selected batters. Empty string if no data.
    Only surfaces the note when occurrences >= 3.
    """
    best_score = 0.0
    best_name  = ""
    for other in others:
        if other == player:
            continue
        data = lookup.get((player, other)) or lookup.get((other, player))
        if data and data["occurrences"] >= 3:
            sc = data["occurrences"] * data["avg_sr"] / 15.0
            if sc > best_score:
                best_score = sc
                best_name  = other
    if best_name:
        data = lookup.get((player, best_name)) or lookup.get((best_name, player))
        sr   = data["avg_sr"]
        occ  = int(data["occurrences"])
        return f"pairs well w/ {best_name.split()[-1]} (SR {sr:.0f}, {occ} times)"
    return ""


def _is_left_hand(p: "ScoredPlayer") -> bool:
    return "left" in p.batting_style.lower()


def _load_finisher_flags(
    players:    list[str],
    stats_path: Path | None = None,
) -> set[str]:
    """
    Identify 'finisher' archetype players who bat best at death overs (positions 6-7).

    Criteria (all must be met):
      - death_bat_sr >= 175 in career death-phase data (elite death striker)
      - death_bat_balls >= 100 (meaningful sample — at least ~17 death innings)
      - death_bat_balls > pp_bat_balls (bats more at death than PP confirms the
        role; openers like Fakhar have pp_balls >> death_balls and are excluded)

    Returns a set of player names that qualify as finishers.
    Finishers should not be placed at batting positions 3, 4, or 5 when
    there are better-suited top-order batters available.
    """
    flags: set[str] = set()
    path = stats_path or STATS_PATH
    try:
        df = pd.read_parquet(path)
        death_rows = df[(df["season"] == 0) & (df["phase"] == "death")]
        pp_rows    = df[(df["season"] == 0) & (df["phase"] == "powerplay")]
    except Exception:
        return flags

    for p in players:
        d_row  = death_rows[death_rows["player_name"] == p]
        pp_row = pp_rows[pp_rows["player_name"] == p]
        if d_row.empty:
            continue
        d = d_row.iloc[0]
        death_sr    = float(d.get("bat_sr",    0.0) or 0.0)
        death_balls = float(d.get("bat_balls", 0.0) or 0.0)

        pp_balls = 0.0
        if not pp_row.empty:
            pp_balls = float(pp_row.iloc[0].get("bat_balls", 0.0) or 0.0)

        # Must bat predominantly in death overs (not an opener with rare death cameos)
        if death_sr >= 175 and death_balls >= 100 and death_balls > pp_balls:
            flags.add(p)

    return flags


def _optimize_batting_order(
    non_bowlers: list["ScoredPlayer"],
    lookup:      dict[tuple[str, str], dict],
    finishers:   "set[str] | None" = None,
) -> list["ScoredPlayer"]:
    """
    Refine the batting order of non-bowler players using partnership history
    and left-hand/right-hand alternation tactics.

    Step 1 — Opening pair selection:
        From the top-5 candidates by score, find the pair with the highest
        partnership score (occurrences × avg_sr). If no meaningful partnership
        exists (score < 5), keep the top-2 by score as openers.

    Step 2 — Middle-order adjacency sweeps (partnership):
        Run 2 passes over positions 3-onward. Swap adjacent players if the
        swap improves the total adjacent-pair partnership chain score.

    Step 3 — LH/RH alternation pass (positions 3-6 only):
        In real T20 cricket, alternating left-hand and right-hand batters
        disrupts the opposition's field placements and forces bowling changes.
        For positions 3-6, try swapping a pair of same-handed adjacent batters
        with a different-handed player nearby, provided:
          - The swap doesn't reduce the chain partnership score by more than 3.
          - Both players' scores are within 8 points (avoid demoting a clearly
            superior player for a minor tactical gain).
        Openers (slots 0-1) and tail (slot 7+) are never touched.
    """
    if len(non_bowlers) < 2:
        return non_bowlers

    _finishers = finishers or set()
    ordered = list(non_bowlers)  # already sorted by score descending

    # --- Step 1: opening pair ---
    # Exclude finisher-archetype players from opening pair candidates.
    # They belong at 6-7 and shouldn't open even if they score highly overall.
    OPEN_CANDS = min(5, len(ordered))
    candidates = [p for p in ordered[:OPEN_CANDS] if p.player_name not in _finishers]
    if len(candidates) < 2:
        candidates = ordered[:OPEN_CANDS]  # fallback if all top-5 are finishers
    best_pair_score = -1.0
    best_i_name, best_j_name = candidates[0].player_name, candidates[1].player_name

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            sc = _partnership_score(candidates[i].player_name, candidates[j].player_name, lookup)
            if sc > best_pair_score:
                best_pair_score = sc
                best_i_name = candidates[i].player_name
                best_j_name = candidates[j].player_name

    if best_pair_score >= 5.0:
        # Reorder so the best opening pair goes to positions 0-1
        names_in_ordered = [p.player_name for p in ordered]
        if best_i_name in names_in_ordered and best_j_name in names_in_ordered:
            pi_idx = names_in_ordered.index(best_i_name)
            pj_idx = names_in_ordered.index(best_j_name)
            if pi_idx != 0 or pj_idx != 1:
                p_i = ordered.pop(max(pi_idx, pj_idx))
                p_j = ordered.pop(min(pi_idx, pj_idx) if max(pi_idx, pj_idx) > min(pi_idx, pj_idx) else min(pi_idx, pj_idx))
                # Insert in score order (higher scorer opens)
                pair = sorted([p_i, p_j], key=lambda x: x.score, reverse=True)
                ordered.insert(0, pair[1])
                ordered.insert(0, pair[0])

    # --- Step 2: middle-order adjacency sweeps (partnership) ---
    for _ in range(2):
        for idx in range(2, len(ordered) - 1):
            a, b = ordered[idx], ordered[idx + 1]
            before = _partnership_score(
                ordered[idx - 1].player_name, a.player_name, lookup
            ) + _partnership_score(a.player_name, b.player_name, lookup)
            after = _partnership_score(
                ordered[idx - 1].player_name, b.player_name, lookup
            ) + _partnership_score(b.player_name, a.player_name, lookup)
            if after > before:
                ordered[idx], ordered[idx + 1] = ordered[idx + 1], ordered[idx]

    # --- Step 3: LH/RH alternation (positions 3-6 = indices 2-5) ---
    # Only try swaps within the middle-order window where the tactical benefit
    # of disrupting opposition field placements outweighs the cost of
    # reordering by score.
    MID_START = 2
    MID_END   = min(6, len(ordered) - 1)  # index upper bound (exclusive)
    SCORE_TOLERANCE   = 8.0    # allow demoting a player by at most 8 score points
    PARTNERSHIP_FLOOR = -3.0   # allow losing at most 3 partnership chain points

    for idx in range(MID_START, MID_END - 1):
        curr = ordered[idx]
        nxt  = ordered[idx + 1]

        # Skip if they already alternate
        if _is_left_hand(curr) != _is_left_hand(nxt):
            continue

        # Both same-handed — look one slot further for an alternation candidate
        # (swap nxt with the player two slots ahead if that player is different hand)
        for alt_idx in range(idx + 2, min(idx + 4, len(ordered))):
            alt = ordered[alt_idx]
            if _is_left_hand(alt) == _is_left_hand(curr):
                continue  # same hand — doesn't help

            # Check score tolerance: don't demote alt if they score much worse than nxt
            if nxt.score - alt.score > SCORE_TOLERANCE:
                break  # too much score gap, stop looking

            # Check partnership impact: compute chain before vs after swap
            prev_name = ordered[idx - 1].player_name if idx > 0 else ""
            chain_before = (
                (_partnership_score(prev_name, nxt.player_name, lookup)  if prev_name else 0)
                + _partnership_score(nxt.player_name, curr.player_name, lookup)
            )
            chain_after = (
                (_partnership_score(prev_name, alt.player_name, lookup)  if prev_name else 0)
                + _partnership_score(alt.player_name, curr.player_name, lookup)
            )
            if chain_after - chain_before >= PARTNERSHIP_FLOOR:
                # Acceptable — perform the swap
                ordered[idx + 1], ordered[alt_idx] = ordered[alt_idx], ordered[idx + 1]
            break  # only try the nearest suitable candidate

    return ordered


# ---------------------------------------------------------------------------
# BATTING ORDER ASSIGNMENT
# ---------------------------------------------------------------------------

def _assign_batting_order(
    selected:           list[ScoredPlayer],
    partnership_lookup: dict | None = None,
    finisher_flags:     set[str] | None = None,
) -> list[XiPlayer]:
    """
    Assign batting positions 1-11 using role, partnership history, and archetype.

    Non-bowlers are ordered by _optimize_batting_order() which:
      1. Selects the best known opening pair from the top-5 candidates.
      2. Runs adjacency sweeps to nudge known batting partners together.
      3. LH/RH alternation pass (positions 3-6).

    Post-processing constraints:
      WK position: keepers who are not top-2 scorers are pushed to position 5.
      Finisher archetype: players with elite death-SR and low avg are pushed to
        position 6-7, freeing positions 3-5 for anchor/top-order batters.

    Pure bowlers are placed at the bottom, ordered by score descending.
    """
    lookup   = partnership_lookup or {}
    finishers = finisher_flags or set()
    batters = sorted([p for p in selected if not p.is_bowler], key=lambda x: x.score, reverse=True)
    bowlers = sorted([p for p in selected if p.is_bowler and not p.is_allrounder],
                     key=lambda x: x.score, reverse=True)

    # Partnership-optimised ordering for batting positions
    ordered_batters = _optimize_batting_order(batters, lookup, finishers=finishers)

    # Step A — Finisher archetype: players with elite death-SR (>=175) and low avg
    # belong at positions 6-7 regardless of their overall score rank.
    # Run this BEFORE the WK fix so that the keeper check sees the post-finisher order.
    if finishers:
        non_fin = [p for p in ordered_batters if p.player_name not in finishers]
        fin_bat = sorted([p for p in ordered_batters if p.player_name in finishers],
                         key=lambda x: x.score, reverse=True)
        if fin_bat:
            FINISHER_INSERT = min(5, len(non_fin))  # position 6 = index 5
            ordered_batters = non_fin[:FINISHER_INSERT] + fin_bat + non_fin[FINISHER_INSERT:]

    # Step B — WK position control: keepers who aren't top-2 scorers among non-bowlers
    # should NOT bat at #3 or #4. They belong at #5 (middle-order anchor).
    # Run AFTER finisher fix so that keepers displaced by finishers are correctly placed.
    # Exception: genuine opener WK (e.g. Rizwan) who ARE the top scorer remain as openers.
    if len(ordered_batters) >= 5:
        nb_scores = sorted([p.score for p in ordered_batters], reverse=True)
        top2_floor = nb_scores[min(1, len(nb_scores) - 1)]  # 2nd-highest score
        for ki in range(len(ordered_batters) - 1, -1, -1):
            p = ordered_batters[ki]
            if p.is_keeper and ki in (2, 3) and p.score < top2_floor - 0.01:
                # Move this keeper from position 3/4 → position 5
                ordered_batters.pop(ki)
                ordered_batters.insert(min(4, len(ordered_batters)), p)
                break  # typically only one keeper

    ordered = ordered_batters + bowlers

    # Build partner note lookup for key_stat
    batter_names = [p.player_name for p in ordered_batters]
    result: list[XiPlayer] = []

    for pos, p in enumerate(ordered, start=1):
        form_badge = f" · {p.form_tag}" if p.form_tag else ""

        if p.is_bowler:
            key_stat = f"Bowler — score {p.score:.0f}{form_badge}"
        elif p.is_keeper:
            # Add partner note for keepers too (they bat)
            partner_note = _best_partner_note(p.player_name, batter_names, lookup)
            partner_tag  = f" · {partner_note}" if partner_note else ""
            key_stat = f"WK — score {p.score:.0f}{form_badge}{partner_tag}"
        else:
            hand = "Left" if "left" in p.batting_style.lower() else "Right"
            partner_note = _best_partner_note(p.player_name, batter_names, lookup)
            partner_tag  = f" · {partner_note}" if partner_note else ""
            key_stat = f"{hand}-hand — score {p.score:.0f}{form_badge}{partner_tag}"

        result.append(XiPlayer(
            batting_position = pos,
            player_name      = p.player_name,
            role             = p.role,
            score            = p.score,
            key_stat         = key_stat,
            model_source     = p.model_source,
        ))

    return result


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# SQUAD VALIDATION HELPER
# ---------------------------------------------------------------------------

def validate_squad(
    squad:      list[str],
    team:       str = "",
    meta_path:  str | None = None,
) -> dict:
    """
    Validate a squad list against the player index before engine processing.
    Catches ghost players (names not in index) at the entry point so they
    cannot silently corrupt downstream results.

    Returns a dict:
      {
        "valid":       bool,
        "known":       list[str],   # names found in player index
        "unknown":     list[str],   # names NOT in player index
        "warnings":    list[str],   # non-fatal issues (e.g. no emerging player)
        "errors":      list[str],   # fatal issues (< 11 players, no keeper, etc.)
      }
    """
    path      = meta_path or str(PLAYER_INDEX)
    meta      = _load_meta(path)
    known     = [p for p in squad if p in meta]
    unknown   = [p for p in squad if p not in meta]

    warnings: list[str] = []
    errors:   list[str] = []

    if unknown:
        errors.append(
            f"{len(unknown)} player(s) not found in player index: {unknown}. "
            "Check spelling or add them to player_index_2026_enriched.csv."
        )

    if len(known) < XI_SIZE:
        errors.append(
            f"Only {len(known)} recognised players — need at least {XI_SIZE} to select a valid XI."
        )

    overseas = sum(1 for p in known if meta[p].get("is_overseas", False))
    if overseas > MAX_OVERSEAS:
        errors.append(f"Squad has {overseas} overseas players — PSL cap is {MAX_OVERSEAS}.")

    emerging = sum(1 for p in known if meta[p].get("is_emerging", False))
    if emerging == 0:
        warnings.append(
            "No emerging (U23 uncapped) player in squad — PSL rules require exactly 1 in the XI."
        )

    keepers = sum(
        1 for p in known
        if meta[p].get("primary_role", "") in ROLE_KEEPER
    )
    if keepers == 0:
        errors.append("No wicketkeeper in squad.")

    bowlers = sum(
        1 for p in known
        if meta[p].get("primary_role", "") in ROLE_BOWLER | ROLE_ALLROUND
    )
    if bowlers < MIN_BOWLERS:
        warnings.append(
            f"Only {bowlers} genuine bowlers/all-rounders — may struggle to cover 20 overs."
        )

    return {
        "valid":   len(errors) == 0,
        "known":   known,
        "unknown": unknown,
        "warnings": warnings,
        "errors":   errors,
    }


# ---------------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------------

def select_xi(
    squad:                  list[str],
    venue:                  str,
    weather:                WeatherImpact,
    innings:                int = 1,
    model_path:             Optional[Path] = None,
    player_index_path:      Optional[Path] = None,
    stats_path:             Optional[Path] = None,
    forced_players:         Optional[list[str]] = None,
    opposition_lh_pct:      float = 33.0,
    opposition_batters:     Optional[list[str]] = None,
    opposition_spin_economies: Optional[dict[str, float]] = None,
) -> list[XiOption]:
    """
    Select the best Playing 11 from a squad, returning 3 alternatives.

    Args:
        squad:    16-18 player names (must be in player_index.csv)
        venue:    Match venue name
        weather:  WeatherImpact (modifies spinner/pace scores)
        innings:  1 (batting first) or 2 (chasing) — affects scoring
        model_path: Override for xi_scorer.pkl

    Returns:
        [OptionA (primary), OptionB (spin-heavy), OptionC (pace-heavy)]
    """
    # Validate squad against player index — surface unknown names early
    _vr = validate_squad(squad)
    if _vr["unknown"]:
        import warnings as _warnings
        _warnings.warn(
            f"select_xi: {len(_vr['unknown'])} player(s) not in index — "
            f"they will be scored with defaults: {_vr['unknown']}",
            stacklevel=2,
        )

    if len(squad) < 11:
        raise ValueError(
            f"select_xi() requires at least 11 players in the squad, got {len(squad)}. "
            f"A PSL squad is typically 16-18 players."
        )

    mp  = str(model_path or MODEL_PATH)
    pi  = str(player_index_path or (PLAYER_INDEX if PLAYER_INDEX.exists() else PLAYER_INDEX_FALLBACK))
    sp  = Path(stats_path) if stats_path else STATS_PATH

    from models.train_xi_scorer import load_model
    # Cache the loaded model payload in the module — loading TabNet + XGBoost
    # takes 3-4s on first call; subsequent calls are instant.
    global _CACHED_MODEL_PAYLOAD, _CACHED_MODEL_KEY
    _cache_key = mp
    if _CACHED_MODEL_PAYLOAD is None or _CACHED_MODEL_KEY != _cache_key:
        _CACHED_MODEL_PAYLOAD = load_model(mp)
        _CACHED_MODEL_KEY = _cache_key
    payload      = _CACHED_MODEL_PAYLOAD
    meta         = _load_meta(pi)
    partnerships = _load_partnerships()

    # Score the squad (form + matchup bonuses applied inside)
    scored = _score_squad(
        squad,
        venue,
        innings,
        weather,
        meta,
        payload,
        stats_path          = sp,
        opp_lh_pct          = opposition_lh_pct,
        opp_batters         = opposition_batters or [],
        opp_spin_economies  = opposition_spin_economies,
    )

    options: list[XiOption] = []
    configs = [
        # Option A: balanced selection — highest composite score
        ("Option A", "Primary XI",                    False, False),
        # Option B: max spinners — for dry/turning tracks or vs spin-vulnerable batting
        ("Option B", "Dry pitch / Spin-friendly",     True,  False),
        # Option C: max pace — for seaming tracks or vs pace-dependent batting lineups
        ("Option C", "Seaming track / Pace-heavy",    False, True),
    ]

    finisher_flags = _load_finisher_flags(squad, stats_path=sp)

    for label, desc, prefer_spin, prefer_pace in configs:
        selected = _greedy_select(scored, prefer_spin=prefer_spin, prefer_pace=prefer_pace,
                                  forced_players=forced_players, venue=venue)
        xi       = _assign_batting_order(selected, partnership_lookup=partnerships,
                                         finisher_flags=finisher_flags)

        overseas = sum(1 for p in selected if p.is_overseas)
        bowlers  = sum(1 for p in selected if p.is_bowler or p.is_allrounder)
        emerging = sum(1 for p in selected if p.is_emerging)
        # Max overs coverage (each bowler 4, allrounder 2)
        coverage = sum(
            4 if p.is_bowler else 2
            for p in selected
            if p.is_bowler or p.is_allrounder
        )
        constraint_note = (
            f"{overseas} overseas / {XI_SIZE - overseas} local  |  "
            f"{emerging} emerging  |  "
            f"{bowlers} bowlers  |  ~{coverage} overs coverage"
        )
        total_score = sum(p.score for p in selected)

        options.append(XiOption(
            label           = label,
            description     = desc,
            players         = xi,
            overseas_count  = overseas,
            bowler_count    = bowlers,
            constraint_note = constraint_note,
            total_score     = round(total_score, 1),
        ))

    return options


# ---------------------------------------------------------------------------
# QUICK SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from utils.situation import WeatherImpact

    # PSL 2026 Lahore Qalandars — 14-player subset of verified 20-man squad
    # 4 overseas (Sikandar Raza, Mustafizur Rahman, Dasun Shanaka, Dunith Wellalage)
    # 2 emerging (Ubaid Shah, Shahab Khan)
    lahore_squad = [
        "Fakhar Zaman", "Abdullah Shafique", "Tayyab Tahir",
        "Sikandar Raza", "Hussain Talat", "Haris Rauf",
        "Usama Mir", "Shaheen Shah Afridi", "Mustafizur Rahman",
        "Dasun Shanaka", "Dunith Wellalage", "Haseebullah",
        "Ubaid Shah", "Shahab Khan",
    ]

    weather = WeatherImpact(
        spinner_penalty    = 0.65,
        swing_bonus        = 1.15,
        pace_bounce_bonus  = 1.08,
        yorker_reliability = 0.90,
        dl_planning_needed = False,
        dew_onset_over     = 12,
        warnings           = [],
    )

    options = select_xi(
        lahore_squad,
        "Gaddafi Stadium, Lahore",
        weather,
        innings=1,
    )

    print(f"\nXI Selection -- Lahore Qalandars @ Gaddafi Stadium")
    print(f"{'='*65}")

    for opt in options:
        print(f"\n  {opt.label}: {opt.description}  (total score: {opt.total_score:.0f})")
        print(f"  {opt.constraint_note}")
        print(f"  {'Pos':<4}  {'Player':<25}  {'Role':<14}  Key stat")
        print(f"  {'-'*70}")
        for p in opt.players:
            print(f"  {p.batting_position:<4}  {p.player_name:<25}  {p.role:<14}  {p.key_stat}")

    print(f"\n{'='*65}\n")
