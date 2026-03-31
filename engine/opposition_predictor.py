# engine/opposition_predictor.py
# Predicts opposition batting order and tendencies for the pre-match brief.
#
# Public API:
#   predict_batting_order(team, venue, our_bowlers, player_index, season) -> OppositionBattingPrediction

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

PROJ_ROOT    = Path(__file__).resolve().parent.parent
PROFILES_PATH = PROJ_ROOT / "data" / "processed" / "opposition_profiles.csv"

# Franchises with no PSL franchise history — batting order built entirely from
# individual player career data, all positions flagged Low confidence.
NEW_FRANCHISES = {"Rawalpindiz", "Hyderabad Kingsmen"}
PLAYER_INDEX  = PROJ_ROOT / "data" / "processed" / "player_index_2026_enriched.csv"
PLAYER_INDEX_FALLBACK = PROJ_ROOT.parent / "player_index_2026_enriched.csv"
MATCHUP_PATH       = PROJ_ROOT / "data" / "processed" / "matchup_matrix.parquet"
BATTING_PROBS_PATH = PROJ_ROOT / "data" / "processed" / "batting_order_probabilities.json"
RECENT_FORM_PATH   = PROJ_ROOT / "data" / "processed" / "recent_form.parquet"


# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class PredictedBatter:
    position:            int
    player_name:         str
    confidence:          str         # "High" | "Medium" | "Low"
    arrival_over_range:  str         # "typically arrives over 8-11"
    batting_style:       str         # "Left-hand" | "Right-hand"
    phase_strength:      str         # e.g. "Powerplay aggressor"
    career_sr:           float
    death_sr:            float
    vs_our_spin_sr:      float
    vs_our_pace_sr:      float
    danger_rating:       str         # "High" | "Medium" | "Low"
    key_note:            str
    position_confidence: str = ""    # "High" | "Medium" | "Low" — based on PSL seasons seen
    position_range:      str = ""    # e.g. "1-3" or "2" — positions seen across seasons


@dataclass
class OppositionBattingPrediction:
    team:                   str
    season:                 int
    predicted_order:        list[PredictedBatter]
    left_hand_count:        int
    aggressive_opener:      bool
    danger_window:          str     # "Most dangerous: positions 4-6 arrive overs 8-14"
    bowling_implications:   list[str]
    powerplay_sr:           float
    death_sr:               float
    vs_spin_economy:        float
    vs_pace_economy:        float
    is_estimated:           bool = False    # True for new teams with no PSL match history


# ---------------------------------------------------------------------------
# LOADERS (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_profiles(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def _load_recent_form_overall(path: str) -> "dict[str, dict]":
    """
    Load recent_form.parquet overall rows (venue=="") into a dict keyed by player_name.
    Returns {} on any error or if file missing.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        df      = pd.read_parquet(p)
        overall = df[df["venue"] == ""]
        result  = {}
        for _, row in overall.iterrows():
            name = row.get("player_name", "")
            if name:
                result[name] = row.to_dict()
        return result
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _load_player_index(path: str) -> dict[str, dict]:
    meta: dict[str, dict] = {}
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
                    "batting_style": row.get("batting_style", "Right-hand bat").strip(),
                    "primary_role":  row.get("primary_role",  "Batsman").strip(),
                    "is_overseas":   row.get("is_overseas",   "False").strip().lower() == "true",
                    # New columns from player_index_2026_enriched.csv
                    "bat_sr_set":       _f("bat_sr_set",       0.0),
                    "bat_sr_chase":     _f("bat_sr_chase",     0.0),
                    "innings_sr_delta": _f("innings_sr_delta", 0.0),
                    "bowl_dot_pct":     _f("bowl_dot_pct",     0.0),
                }
    return meta


@lru_cache(maxsize=1)
def _load_matchup_matrix(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def _load_squad_data(path: str) -> tuple:
    """Returns (team->players list dict, player->primary_role dict) from player_index."""
    teams: dict[str, list[str]] = {}
    roles: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("player_name", "").strip()
            team = row.get("current_team_2026", "").strip()
            role = row.get("primary_role", "Batsman").strip()
            if name and team:
                teams.setdefault(team, []).append(name)
                roles[name] = role
    return teams, roles


def _reload_caches() -> None:
    _load_profiles.cache_clear()
    _load_player_index.cache_clear()
    _load_matchup_matrix.cache_clear()
    _load_squad_data.cache_clear()


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _get_profile(
    profiles: pd.DataFrame,
    team: str,
    season: int,
) -> pd.Series | None:
    """
    Get a recency-weighted profile for the team.

    If season > 0 is explicitly requested, return that season directly.
    Otherwise (season=0 / default), blend the last 3 seasons:
        2025 → 50%  |  2024 → 30%  |  2023 → 20%
    for all numeric stat columns, falling back to career averages for
    columns where recent season data is missing.

    This prevents career aggregates from blending 10 seasons of entirely
    different squads — the 2019 Karachi Kings has zero relevance to 2026.
    """
    # Explicit season request → return it directly (or fall back to career)
    if season != 0:
        row = profiles[(profiles["team"] == team) & (profiles["season"] == season)]
        if not row.empty:
            return row.iloc[0]
        # fall through to recency blend

    NUMERIC_STAT_COLS = [
        "left_hand_top6_pct", "aggressive_opener_pct",
        "powerplay_sr", "middle_sr", "death_sr",
        "vs_spin_economy", "vs_pace_economy", "vs_leftarm_economy",
        "chase_win_pct", "defend_win_pct",
        "powerplay_bowling_economy", "death_bowling_economy",
        "pace_overs_pct", "spin_overs_pct",
    ]

    # These columns can never legitimately be 0.0 — a stored zero means the
    # original value was null/missing and was not handled upstream.
    # Treat 0.0 as missing for these so it doesn't suppress the weighted average.
    NONZERO_COLS = {
        "powerplay_sr", "middle_sr", "death_sr",
        "vs_spin_economy", "vs_pace_economy", "vs_leftarm_economy",
        "powerplay_bowling_economy", "death_bowling_economy",
    }

    SEASON_WEIGHTS = [(2025, 0.50), (2024, 0.30), (2023, 0.20)]

    # Collect available recent-season rows
    available: list[tuple[float, pd.Series]] = []
    for yr, w in SEASON_WEIGHTS:
        r = profiles[(profiles["team"] == team) & (profiles["season"] == yr)]
        if not r.empty:
            available.append((w, r.iloc[0]))

    # Fall back to career if no recent data
    career_row = profiles[(profiles["team"] == team) & (profiles["season"] == 0)]
    career = career_row.iloc[0] if not career_row.empty else None

    if not available:
        return career  # no recent data at all

    # Build a blended Series: for each stat column, weighted average of available seasons
    # Normalise weights to sum to 1.0 across only the seasons that exist
    total_w = sum(w for w, _ in available)
    blended: dict = {}

    # Copy non-numeric fields from the most recent row (or career)
    base_row = available[0][1]  # most recent available
    for col in profiles.columns:
        blended[col] = base_row.get(col)

    # Override stat columns with recency-weighted values
    for col in NUMERIC_STAT_COLS:
        weighted_sum = 0.0
        found_w = 0.0
        for w, row in available:
            v = row.get(col)
            try:
                fv = float(v) if pd.notna(v) else None
            except (ValueError, TypeError):
                fv = None
            # For rate/SR columns, 0.0 is physically impossible and indicates
            # a null that was stored as zero upstream — treat it as missing.
            if fv is not None and not (col in NONZERO_COLS and fv == 0.0):
                weighted_sum += fv * w
                found_w += w
        if found_w > 0:
            blended[col] = round(weighted_sum / found_w, 2)
        elif career is not None:
            blended[col] = career.get(col)  # fall back to career for missing col

    # Season field: mark as 2025 (most recent) so confidence scoring works correctly
    blended["season"] = available[0][1].get("season", 2025)
    blended["is_estimated"] = base_row.get("is_estimated", False)

    return pd.Series(blended)


def _batting_style(player: str, meta: dict[str, dict]) -> str:
    style = meta.get(player, {}).get("batting_style", "Right-hand bat")
    return "Left-hand" if "left" in style.lower() else "Right-hand"


def _arrival_over(position: int) -> str:
    """Estimate typical arrival window based on batting position."""
    windows = {
        1: "opens — arrives over 0",
        2: "opens — arrives over 0",
        3: "arrives over 1-4 (first wicket)",
        4: "typically arrives over 4-8",
        5: "typically arrives over 6-10",
        6: "typically arrives over 8-12",
        7: "typically arrives over 10-14",
        8: "typically arrives over 12-16",
        9: "typically arrives over 14-18",
        10: "typically arrives over 16-19",
        11: "typically arrives over 17-19",
    }
    return windows.get(position, f"arrives over {max(0, position*2 - 4)}-{min(20, position*2)}")


def _phase_strength(position: int, profile) -> str:
    """Classify batter phase role based on position and team profile."""
    pp_sr  = float(profile.get("powerplay_sr", 120) or 120) if profile is not None else 120.0
    dth_sr = float(profile.get("death_sr",     130) or 130) if profile is not None else 130.0

    if position <= 2:
        if pp_sr >= 135:
            return "Powerplay aggressor"
        return "Powerplay opener"
    elif position <= 4:
        return "Top-order anchor"
    elif position <= 6:
        if dth_sr >= 160:
            return "Death finisher"
        return "Middle-order builder"
    else:
        return "Death hitter / lower order"


def _vs_bowler_sr(
    batter: str,
    bowlers: list[str],
    bowl_type: str,         # "pace" or "spin"
    matrix: pd.DataFrame,
    player_meta: dict[str, dict],
) -> float:
    """
    Average SR of this batter against our bowlers of a given type.
    Falls back to 130.0 if no data.
    """
    type_bowlers = [
        b for b in bowlers
        if (bowl_type == "pace" and _is_pace(b, player_meta))
        or (bowl_type == "spin" and _is_spin(b, player_meta))
    ]
    if not type_bowlers:
        return 130.0

    rows = matrix[
        (matrix["batter"] == batter)
        & (matrix["bowler"].isin(type_bowlers))
        & (matrix["balls"] >= 6)
    ]
    if rows.empty:
        return 130.0

    total_runs  = rows["runs"].sum()
    total_balls = rows["balls"].sum()
    return round(total_runs / total_balls * 100, 1) if total_balls > 0 else 130.0


def _is_pace(player: str, meta: dict[str, dict]) -> bool:
    style = meta.get(player, {}).get("bowling_style", "").lower()
    return any(w in style for w in ("fast", "medium", "seam", "swing", "pace"))


def _is_spin(player: str, meta: dict[str, dict]) -> bool:
    style = meta.get(player, {}).get("bowling_style", "").lower()
    return any(w in style for w in ("spin", "off", "leg", "googly", "chinaman", "slow"))


def _danger_rating(
    position:       int,
    death_sr:       float,
    career_sr:      float,
    vs_spin_sr:     float,
    bat_form_score: float = 50.0,
) -> str:
    """
    Phase-aware danger rating.

    Top-order (pos 1-3): career SR drives the rating — they threaten across all phases.
    Lower-order (pos 6+): death SR drives the rating — they are death-phase specialists.
    A #7 finisher with death SR 195 is more dangerous in overs 16-20 than a #2
    with career SR 145, so death SR must dominate for lower-order batters.

    Recent form adjustment (bat_form_score from recent_form.parquet):
      ≥70: upgrade rating one level (Low→Medium, Medium→High)
      <35: downgrade rating one level (High→Medium, Medium→Low)
    """
    score = 0

    # Career SR — relevant across all positions but weighted differently
    if career_sr >= 145:   score += 2
    elif career_sr >= 130: score += 1

    # Top-order position bonus (threat throughout the innings)
    if position <= 2:
        score += 1    # opener bonus (reduced from 2 — position alone ≠ dangerous)
    elif position == 3:
        score += 1    # #3 is a key position but not automatically dangerous

    # Death-phase threat — especially critical for lower-order hitters.
    # A position-6+ batter with SR 175+ is a match-winner in overs 16-20.
    if death_sr >= 185:    score += 3   # elite finisher, game-changing
    elif death_sr >= 165:  score += 2   # strong finisher
    elif death_sr >= 145:  score += 1   # decent death hitter

    # Lower-order batters face limited balls outside death overs — reduce top-order
    # position advantage to avoid inflating danger for batters who rarely score in mid.
    # Exception: genuine death finishers (death SR ≥ 165) are a real threat — no penalty.
    if position >= 6 and score > 0 and death_sr < 165:
        score -= 1   # partial offset: lower-order is phase-limited (not finishers)

    # vs-spin vulnerability adds danger (they will be aggressive against our spinners)
    if vs_spin_sr >= 145: score += 1

    if score >= 4:
        base_rating = "High"
    elif score >= 2:
        base_rating = "Medium"
    else:
        base_rating = "Low"

    # Fix 3.2: Recent form adjustment — only for recognized batting positions (1-6).
    # Tail-enders (7-11) rarely face enough balls outside the death to warrant a
    # full danger-level shift; their form is too volatile over small samples.
    if position <= 6:
        _UPGRADE   = {"Low": "Medium", "Medium": "High", "High": "High"}
        _DOWNGRADE = {"High": "Medium", "Medium": "Low", "Low": "Low"}
        if bat_form_score >= 70:
            return _UPGRADE[base_rating]
        if bat_form_score < 35:
            return _DOWNGRADE[base_rating]
    return base_rating


def _build_key_note(
    player:         str,
    position:       int,
    profile:        pd.Series,
    meta:           dict[str, dict],
    career_sr:      float,
    death_sr:       float,
    vs_spin_sr:     float,
    vs_pace_sr:     float,
    recent_form:    dict | None = None,
) -> str:
    """Generate a one-line tactical note for this batter."""
    batting_style = _batting_style(player, meta)
    notes = []

    if position <= 2 and profile is not None and profile.get("aggressive_opener_pct", 0) >= 50:
        notes.append(f"aggressive opener (PP SR > 130 in {profile.get('aggressive_opener_pct', 0):.0f}% of innings)")
    if death_sr >= 165:
        notes.append(f"elite death hitter (SR {death_sr:.0f} in death overs)")
    if vs_spin_sr >= 150:
        notes.append(f"attacks spin (SR {vs_spin_sr:.0f} vs spinners)")
    elif vs_spin_sr <= 100:
        notes.append(f"vulnerable to spin (SR {vs_spin_sr:.0f} vs spinners)")
    if vs_pace_sr <= 100 and career_sr >= 120:
        notes.append(f"struggles vs pace (SR {vs_pace_sr:.0f})")
    if batting_style == "Left-hand":
        notes.append("left-hand bat")

    base = ("; ".join(notes).capitalize() + ".") if notes else f"Career PSL SR {career_sr:.0f}."

    # Append recent form context when available
    if recent_form:
        r_avg = recent_form.get("bat_avg", 0.0)
        r_sr  = recent_form.get("bat_sr",  0.0)
        r_inn = recent_form.get("bat_innings", 0)
        if r_inn >= 3 and r_avg > 0:
            base += f" Current form: {r_avg:.0f} avg, {r_sr:.0f} SR last 10 T20s."

    return base


# ---------------------------------------------------------------------------
# MAIN PREDICTION FUNCTION
# ---------------------------------------------------------------------------

def predict_batting_order(
    team:          str,
    venue:         str,
    our_bowlers:   list[str],
    season:        int = 0,
    profiles_path: Optional[Path] = None,
    player_index_path: Optional[Path] = None,
    matchup_path:  Optional[Path] = None,
) -> OppositionBattingPrediction:
    """
    Predict the opposition batting order and tendencies.

    Args:
        team:         Opposition team name (must match opposition_profiles.csv)
        venue:        Match venue (currently informational)
        our_bowlers:  List of our squad bowler names (for vs_spin/pace SR lookup)
        season:       PSL season year (0 = career average). Falls back to career if not found.

    Returns:
        OppositionBattingPrediction with positions 1-8 (typical playing XI batting positions)
    """
    # Resolve paths
    pp     = str(profiles_path    or PROFILES_PATH)
    pi     = str(player_index_path or (PLAYER_INDEX if PLAYER_INDEX.exists() else PLAYER_INDEX_FALLBACK))
    mp     = str(matchup_path     or MATCHUP_PATH)

    profiles = _load_profiles(pp)
    meta     = _load_player_index(pi)
    matrix   = _load_matchup_matrix(mp)
    team_squads, player_roles = _load_squad_data(pi)

    # New franchises: skip team-level history lookup entirely — build from
    # individual player career positions with universal Low confidence.
    is_new_franchise = team in NEW_FRANCHISES
    if is_new_franchise:
        profile = None
    else:
        profile = _get_profile(profiles, team, season)

    if profile is None and not is_new_franchise:
        # Unknown team — return a minimal placeholder
        return OppositionBattingPrediction(
            team                 = team,
            season               = season,
            predicted_order      = [],
            left_hand_count      = 0,
            aggressive_opener    = False,
            danger_window        = "No historical data for this team.",
            bowling_implications = ["Insufficient data — rely on manual scouting."],
            powerplay_sr         = 120.0,
            death_sr             = 130.0,
            vs_spin_economy      = 7.5,
            vs_pace_economy      = 8.0,
            is_estimated         = False,
        )

    # Parse batting order JSON (skipped for new franchises — no history exists)
    batting_order_raw: dict[str, str] = {}
    if profile is not None:
        try:
            batting_order_raw = json.loads(str(profile.get("typical_batting_order") or "{}"))
        except (json.JSONDecodeError, TypeError):
            pass

    # Sort by position key — guard against non-integer keys from malformed JSON
    try:
        order_items = sorted(batting_order_raw.items(), key=lambda x: int(x[0]))
    except (ValueError, TypeError):
        order_items = []   # malformed batting order — squad filter below will fill gaps

    # --- Dynamic Squad Prediction: Use the ML Engine ---
    # Generate a fresh XI using the ML engine instead of purely historic JSON,
    # ensuring the 4-overseas rule and partnership algorithms are enforced equally for opponents.
    valid_players = list(team_squads.get(team, []))
    gap_filled_set: set[int] = set()
    if valid_players:
        from engine.xi_selector import select_xi
        from utils.situation import WeatherImpact
        # Assume a neutral weather state to just get the base best-XI
        base_weather = WeatherImpact(0.0, 1.0, 1.0, 1.0, False, 20, [])
        try:
            options = select_xi(squad=valid_players, venue=venue, weather=base_weather, innings=1)
            if options and options[0].players:
                # Rebuild order_items completely from the optimal ML generated XI
                order_items = [
                    (str(p.batting_position), p.player_name) 
                    for p in options[0].players
                ]
            else:
                order_items = []
        except Exception:
            # Fallback to historic order items if the engine errors
            order_items = [(pos, p) for pos, p in order_items if p in set(valid_players)]
        
        order_items.sort(key=lambda x: int(x[0]))
        order_items = order_items[:8]   # limit to top-8 batting positions

    # Determine confidence based on how recent the season data is
    if is_new_franchise or profile is None:
        profile_season  = 0
        base_confidence = "Low"   # no franchise history
    else:
        profile_season = int(profile.get("season", 0))
        if profile_season == 0:
            base_confidence = "Medium"   # career data
        elif profile_season >= 2023:
            base_confidence = "High"
        else:
            base_confidence = "Low"

    def _position_confidence(pos: int, base: str) -> str:
        """
        Cap confidence by batting position.
        T20 batting orders are volatile — positions 6-11 change every match
        based on match state, promotions, and tactical decisions.
        Only openers (1-2) can realistically carry 'High' confidence.
        """
        if pos <= 2:
            return base                 # openers: data-driven confidence applies
        elif pos <= 5:
            # Middle order: cap at Medium — teams promote/demote based on game state
            return "Medium" if base == "High" else base
        else:
            # Positions 6-11: always Low — lower order is inherently unpredictable
            return "Low"

    # Extract phase SRs from career stats in player_stats if available
    stats_path = PROJ_ROOT / "data" / "processed" / "player_stats.parquet"
    try:
        pstats = pd.read_parquet(stats_path)
        career_stats = pstats[pstats["season"] == 0]
    except Exception:
        career_stats = pd.DataFrame()

    def _player_sr(player: str, phase: str = "overall") -> float:
        if career_stats.empty:
            return 120.0
        row = career_stats[
            (career_stats["player_name"] == player) & (career_stats["phase"] == phase)
        ]
        if row.empty:
            return 120.0
        v = row.iloc[0].get("bat_sr")
        return float(v) if pd.notna(v) else 120.0

    # Load batting order probabilities (Fix 2 — position confidence from PSL seasons)
    _bat_probs: dict = {}
    _bp_path = BATTING_PROBS_PATH
    if _bp_path.exists():
        try:
            with open(_bp_path, encoding="utf-8") as _bpf:
                _bat_probs = json.load(_bpf)
        except Exception as _bpe:
            import warnings as _bpw
            _bpw.warn(
                f"Could not load batting_order_probabilities.json: {_bpe}. "
                "Position confidence fields will be empty.",
                UserWarning,
                stacklevel=2,
            )
    else:
        import warnings as _bpw2
        _bpw2.warn(
            f"batting_order_probabilities.json not found at {_bp_path}. "
            "Run scripts/build_batting_probabilities.py to generate it. "
            "Position confidence fields will be empty.",
            UserWarning,
            stacklevel=2,
        )

    # Load recent form data (best-effort; empty dict if parquet unavailable)
    _rf_map = _load_recent_form_overall(str(RECENT_FORM_PATH))

    # Build predicted batters
    predicted: list[PredictedBatter] = []
    left_hand_count = 0

    for pos_str, player in order_items:
        pos      = int(pos_str)
        bat_sty  = _batting_style(player, meta)
        career_sr = _player_sr(player, "overall")
        death_sr  = _player_sr(player, "death")
        if death_sr == 120.0 and profile is not None:  # fallback: use profile death_sr proxy
            death_sr = float(profile.get("death_sr", 130) or 130)

        vs_spin_sr = _vs_bowler_sr(player, our_bowlers, "spin", matrix, meta)
        vs_pace_sr = _vs_bowler_sr(player, our_bowlers, "pace", matrix, meta)

        if bat_sty == "Left-hand":
            left_hand_count += 1

        player_rf     = _rf_map.get(player)
        bat_form_score = float(player_rf.get("bat_form_score", 50.0)) if player_rf else 50.0

        danger = _danger_rating(pos, death_sr, career_sr, vs_spin_sr, bat_form_score)
        note   = _build_key_note(
            player, pos, profile, meta,
            career_sr, death_sr, vs_spin_sr, vs_pace_sr, player_rf
        )

        _player_probs  = _bat_probs.get(team, {}).get(player, {})
        _pos_conf  = _player_probs.get("position_confidence", "")
        _pos_range = _player_probs.get("position_range", "")
        if _pos_conf and pos > 5:
            _pos_conf = "Low"
        if not _pos_conf:
            _pos_conf = "Low"
            _pos_range = "unknown"

        # Fix 3.6: mark gap-filled positions so analysts know this is an estimate,
        # not a data-backed position.
        if pos in gap_filled_set:
            _pos_conf  = "Low"
            _pos_range = "estimated"
            note = f"[Position estimated — no PSL data at #{pos}] " + note

        # New-franchise override: all positions are Low confidence; prepend notice.
        if is_new_franchise:
            _pos_conf  = "Low"
            _pos_range = "estimated"
            note = "[New PSL franchise — position based on T20 career data only] " + note

        predicted.append(PredictedBatter(
            position             = pos,
            player_name          = player,
            confidence           = _position_confidence(pos, base_confidence),
            arrival_over_range   = _arrival_over(pos),
            batting_style        = bat_sty,
            phase_strength       = _phase_strength(pos, profile),
            career_sr            = career_sr,
            death_sr             = death_sr,
            vs_our_spin_sr       = vs_spin_sr,
            vs_our_pace_sr       = vs_pace_sr,
            danger_rating        = danger,
            key_note             = note,
            position_confidence  = _pos_conf,
            position_range       = _pos_range,
        ))

    # Identify danger window (positions rated High danger)
    high_danger = [p for p in predicted if p.danger_rating == "High"]
    if high_danger:
        positions = [str(p.position) for p in high_danger]
        avg_arrival = sum(
            max(0, p.position * 2 - 2) for p in high_danger
        ) / len(high_danger)
        danger_window = (
            f"Danger positions: {', '.join(positions)} — "
            f"typically arriving around over {int(avg_arrival)}-{int(avg_arrival)+4}"
        )
    elif predicted:
        danger_window = "No standout danger batter identified — consistent lineup."
    else:
        danger_window = "No prediction data available."

    # Bowling implications
    implications: list[str] = []

    pp_sr     = float(profile.get("powerplay_sr",    120) or 120) if profile is not None else 120.0
    dth_sr    = float(profile.get("death_sr",        130) or 130) if profile is not None else 130.0
    vs_spin   = float(profile.get("vs_spin_economy",   8) or 8)  if profile is not None else 8.0
    vs_pace   = float(profile.get("vs_pace_economy",   8) or 8)  if profile is not None else 8.0

    if pp_sr >= 135:
        implications.append(
            f"Aggressive powerplay ({pp_sr:.0f} SR) — lead with wicket-taking pace in overs 1-3."
        )
    elif pp_sr < 115:
        implications.append(
            f"Tentative powerplay ({pp_sr:.0f} SR) — apply pressure early, they will struggle."
        )

    if dth_sr >= 165:
        implications.append(
            f"Dangerous in death ({dth_sr:.0f} SR) — save your best death bowler for overs 18-20."
        )

    if vs_spin <= 7.0:
        implications.append(
            f"Strong vs spin (economy {vs_spin:.1f}) — limit spin overs, focus on pace."
        )
    elif vs_spin >= 8.5:
        implications.append(
            f"Struggles vs spin (economy {vs_spin:.1f} conceded) — use spin aggressively in middle overs."
        )

    left_hand_pct = float(profile.get("left_hand_top6_pct", 0) or 0) if profile is not None else 0.0
    if left_hand_pct >= 40:
        implications.append(
            f"{left_hand_pct:.0f}% left-handers in top 6 — left-arm options are a key weapon."
        )

    if is_new_franchise:
        implications.insert(0,
            "New PSL franchise — position based on T20 career data only. "
            "No franchise-level historical batting order available."
        )
    if not implications:
        implications.append("Balanced batting lineup — no specific bowling adjustments needed.")

    return OppositionBattingPrediction(
        team                = team,
        season              = profile_season,
        predicted_order     = predicted,
        left_hand_count     = left_hand_count,
        aggressive_opener   = bool(profile.get("aggressive_opener_pct", 0) >= 50) if profile is not None else False,
        danger_window       = danger_window,
        bowling_implications= implications,
        powerplay_sr        = pp_sr,
        death_sr            = dth_sr,
        vs_spin_economy     = vs_spin,
        vs_pace_economy     = vs_pace,
        is_estimated        = True if is_new_franchise else bool(profile.get("is_estimated", False)),
    )


# ---------------------------------------------------------------------------
# QUICK SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    lahore_bowlers = [
        "Shaheen Shah Afridi", "Haris Rauf", "Usama Mir",
        "Mustafizur Rahman", "Sikandar Raza",
    ]

    pred = predict_batting_order(
        team        = "Karachi Kings",
        venue       = "National Stadium, Karachi",
        our_bowlers = lahore_bowlers,
        season      = 0,
    )

    print(f"\nOpposition Batting Order Prediction: {pred.team} (season={pred.season})")
    print(f"{'='*70}")
    print(f"  PP SR: {pred.powerplay_sr:.0f}  |  Death SR: {pred.death_sr:.0f}  "
          f"|  Left-handers in top 6: {pred.left_hand_count}")
    print(f"  Aggressive opener: {pred.aggressive_opener}")
    print(f"  {pred.danger_window}")
    print(f"\n  {'Pos':<4}  {'Player':<25}  {'Conf':<8}  {'Danger':<8}  {'Arrival'}")
    print(f"  {'-'*75}")
    for b in pred.predicted_order:
        print(
            f"  {b.position:<4}  {b.player_name:<25}  {b.confidence:<8}  "
            f"{b.danger_rating:<8}  {b.arrival_over_range}"
        )
        print(f"        {b.key_note}")

    print(f"\n  Bowling implications:")
    for imp in pred.bowling_implications:
        print(f"    - {imp}")
