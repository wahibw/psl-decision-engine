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

from utils.situation import WeatherImpact

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

PROJ_ROOT    = Path(__file__).resolve().parent.parent
STATS_PATH   = PROJ_ROOT / "data" / "processed" / "player_stats.parquet"
PLAYER_INDEX = PROJ_ROOT / "data" / "processed" / "player_index.csv"
PLAYER_INDEX_FALLBACK = PROJ_ROOT.parent / "player_index.csv"
OPP_PROFILES = PROJ_ROOT / "data" / "processed" / "opposition_profiles.csv"

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

MAX_OVERS_PER_BOWLER = 4
TOTAL_OVERS          = 20

PHASE_LABEL = {
    0:  "PP", 1:  "PP", 2:  "PP", 3:  "PP", 4:  "PP", 5:  "PP",
    6:  "Mid",7:  "Mid",8:  "Mid",9:  "Mid",10: "Mid",11: "Mid",
    12: "Mid",13: "Mid",14: "Mid",15: "Death",16: "Death",
    17: "Death",18: "Death",19: "Death",
}

PHASE_NAME = {
    "PP":    "powerplay",
    "Mid":   "middle",
    "Death": "death",
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
            meta[name] = {
                "primary_role":  row.get("primary_role", "Batsman").strip(),
                "bowling_style": row.get("bowling_style", "").strip(),
                "is_pace":  any(w in style for w in ("fast","medium","seam","swing","pace")),
                "is_spin":  any(w in style for w in ("spin","off","leg","googly","chinaman","slow")),
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
                "t20_eco":   ipl_eco  if use_ipl else t20_eco,
                "data_tier": 2        if use_ipl else tier,
            }
    return proxy


def _load_opposition_profile(
    opposition_team: str,
    opp_path: Path | None = None,
) -> dict:
    """
    Load career-aggregate opposition profile from opposition_profiles.csv.
    Returns a dict with keys:
        left_hand_top6_pct, vs_spin_economy, vs_pace_economy,
        vs_leftarm_economy, powerplay_sr, death_sr, is_estimated
    Returns sensible neutral defaults if team not found.
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
        return {
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
            "is_estimated":            bool(r.get("is_estimated", True)),
        }
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
        PROXY_OVERS = 12.0
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
        }

    return result


# ---------------------------------------------------------------------------
# PHASE FITNESS SCORING
# ---------------------------------------------------------------------------

def _phase_fitness(
    b:       str,
    phase:   str,
    stats:   dict[str, dict],
    meta:    dict[str, dict],
    weather: WeatherImpact,
    opp:     dict | None = None,
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

    if phase == "PP":
        eco    = s.get("pp_economy",  8.5)
        wpo    = s.get("pp_wkts_po",  0.2)
        dots   = s.get("bowl_dot_pct", 35.0)
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

        # Left-hand heavy batting lineup: left-arm bowlers get extra angle advantage
        if is_left_arm and opp_lh_pct >= 50:
            raw += 6.0

    elif phase == "Death":
        eco    = s.get("death_economy",  9.5)
        wpo    = s.get("death_wkts_po",  0.2)
        dots   = s.get("bowl_dot_pct",  35.0)
        sample = s.get("death_overs",    0.0)

        # Dew degrades spinners at death — raise their effective economy
        if bt == "spin":
            eco = eco / max(0.3, weather.spinner_penalty)

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

    else:  # Mid
        eco    = s.get("mid_economy",  8.0)
        wpo    = s.get("mid_wkts_po",  0.2)
        dots   = s.get("bowl_dot_pct", 35.0)
        sample = s.get("mid_overs",    0.0)

        # Dry/slow conditions (low swing bonus) suit spinners more
        if bt == "spin" and weather.swing_bonus <= 1.05:
            eco = eco * 0.92    # slight bonus

        eco_s  = _norm(12.0 - eco,  0.0, 6.0)
        wpo_s  = _norm(wpo,         0.0, 0.50)
        dot_s  = _norm(dots,        25.0, 58.0)
        raw    = eco_s * 0.50 + wpo_s * 0.30 + dot_s * 0.20

        # Opposition vs spin in middle: the key matchup phase for spinners
        if bt == "spin" and opp_vs_spin_eco < 7.0:
            raw -= 8.0  # they attack spinners hard
        elif bt == "spin" and opp_vs_spin_eco > 9.5:
            raw += 8.0  # they struggle against spin — front-load spinners

        if bt == "pace" and opp_vs_pace_eco < 7.5:
            raw -= 5.0
        elif bt == "pace" and opp_vs_pace_eco > 9.5:
            raw += 5.0

        # Left-arm angle in the middle
        if is_left_arm and opp_lh_pct >= 50:
            raw += 4.0

    # Sample reliability blend: small-sample bowlers converge toward 45
    RELIABLE_OVERS = 20.0
    confidence = min(1.0, sample / RELIABLE_OVERS)
    return round(raw * confidence + 45.0 * (1.0 - confidence), 2)


# ---------------------------------------------------------------------------
# PHASE SPECIALIST CLASSIFICATION
# ---------------------------------------------------------------------------

def _classify_bowlers(
    bowlers:     list[str],
    stats:       dict[str, dict],
    meta:        dict[str, dict],
    weather:     WeatherImpact,
    opp:         dict | None = None,
) -> dict[str, list[str]]:
    """
    Classify each bowler into phases, ranked by _phase_fitness (descending).
    All genuine bowlers appear in all three pools; part-timers only in Mid.
    Spinners are excluded from Death pool if severe dew.
    Opposition profile (opp) adjusts fitness scores when provided.
    """
    pp_bowlers:    list[tuple[str, float]] = []
    mid_bowlers:   list[tuple[str, float]] = []
    death_bowlers: list[tuple[str, float]] = []

    for b in bowlers:
        bt         = _bowl_type(b, meta)
        is_genuine = _is_genuine_bowler(b, meta)

        pp_fit    = _phase_fitness(b, "PP",    stats, meta, weather, opp)
        mid_fit   = _phase_fitness(b, "Mid",   stats, meta, weather, opp)
        death_fit = _phase_fitness(b, "Death", stats, meta, weather, opp)

        # All genuine bowlers enter all phase pools
        if is_genuine:
            pp_bowlers.append((b, pp_fit))
            mid_bowlers.append((b, mid_fit))
            if not (weather.severe_dew and bt == "spin"):
                death_bowlers.append((b, death_fit))

        # Part-timers only enter mid pool (safest phase for them)
        else:
            mid_bowlers.append((b, mid_fit * 0.6))   # discounted fitness

    # Sort descending (highest fitness first)
    pp_bowlers.sort(key=lambda x: x[1], reverse=True)
    mid_bowlers.sort(key=lambda x: x[1], reverse=True)
    death_bowlers.sort(key=lambda x: x[1], reverse=True)

    return {
        "PP":    [b for b, _ in pp_bowlers],
        "Mid":   [b for b, _ in mid_bowlers],
        "Death": [b for b, _ in death_bowlers],
    }


# ---------------------------------------------------------------------------
# PLAN BUILDER
# ---------------------------------------------------------------------------

def generate_bowling_plan(
    our_bowlers:           list[str],
    weather:               WeatherImpact,
    venue:                 str = "",
    opposition_team:       str = "",
    stats_path:            Optional[Path] = None,
    player_index_path:     Optional[Path] = None,
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
    opp   = _load_opposition_profile(opposition_team)

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

    # Phase-specialist classification (opposition-aware)
    phase_pools = _classify_bowlers(genuine + parttimers, stats, meta, weather, opp)

    pp_pool    = phase_pools["PP"]
    mid_pool   = phase_pools["Mid"]
    death_pool = phase_pools["Death"]

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
    #   PP      overs 0-5   (6 overs): opening pair, 3 overs each
    #   Mid-1   overs 6-10  (5 overs): first middle pair (e.g. spinners)
    #   Mid-2   overs 11-14 (4 overs): second middle pair (change-up / returnee)
    #   Death   overs 15-19 (5 overs): death pair, 3+2 overs
    # ------------------------------------------------------------------

    overs_used:  dict[str, int] = {b: 0 for b in our_bowlers}
    # death_reserved[b] = number of overs to hold back from non-death phases
    death_reserved: dict[str, int] = {}
    assignments: dict[int, str] = {}
    backup:      dict[int, str] = {}

    # ------------------------------------------------------------------
    # PRE-PLAN: reserve death overs for top-2 death specialists
    # Both top death bowlers reserve 2 overs each (4 of the 5 death overs).
    # This prevents them from being fully consumed in the middle overs,
    # and ensures the 5th death over goes to the next best available.
    # ------------------------------------------------------------------
    top_death = death_pool[:2]
    for b in top_death:
        death_reserved[b] = 2

    def _available_now(b: str, phase: str) -> bool:
        """Check if b can bowl in this phase without breaking death reservation."""
        used  = overs_used.get(b, 0)
        quota = MAX_OVERS_PER_BOWLER
        if phase != "Death":
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
            end_b = end_a

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

    # PP: one opening pair (best 2 PP fitness), 3 overs each
    _assign_pair(list(range(0, 6)),   "PP",  pp_pool)

    # Middle block 1 (overs 7-11): best mid fitness pair
    _assign_pair(list(range(6, 11)),  "Mid", mid_pool)

    # Middle block 2 (overs 12-15): change-up pair — prefer bowlers not yet used heavily
    # Death specialists' reserved overs prevent them appearing here
    remaining_mid = [b for b in mid_pool + pp_pool if _available_now(b, "Mid")]
    _assign_pair(list(range(11, 15)), "Mid", remaining_mid if remaining_mid else mid_pool)

    # Death: two death specialists, 3+2 overs
    _assign_pair(list(range(15, 20)), "Death", death_pool)

    # ------------------------------------------------------------------
    # Build OverAssignment list
    # ------------------------------------------------------------------
    over_assignments: list[OverAssignment] = []
    prev_primary: Optional[str] = None

    for over in range(TOTAL_OVERS):
        primary = assignments.get(over, our_bowlers[0] if our_bowlers else "TBD")
        bkp     = backup.get(over, primary)
        phase   = PHASE_LABEL[over]
        bt      = _bowl_type(primary, meta)
        s       = stats.get(primary, {})
        partner = bkp

        # Reason — distinguish new spell vs continuing spell
        new_spell = (primary != prev_primary)
        if phase == "PP":
            econ   = s.get("pp_economy", 8.5)
            reason = (
                f"New ball — opens with {partner} from other end. PP economy {econ:.1f}."
                if new_spell
                else f"Continuing PP spell — economy {econ:.1f}."
            )
        elif phase == "Death":
            econ   = s.get("death_economy", 9.5)
            reason = (
                f"Death partnership with {partner} — economy {econ:.1f}."
                if new_spell
                else f"Continuing death spell — economy {econ:.1f}."
            )
        else:
            econ   = s.get("mid_economy", 7.5)
            reason = (
                f"Middle spell — pair with {partner}. Economy {econ:.1f}."
                if new_spell
                else f"Continuing middle spell — economy {econ:.1f}."
            )

        # Weather note
        w_note = ""
        if phase == "Death" and bt == "spin" and weather.dew_onset_over <= over:
            w_note = (
                f"Dew warning — spinner risky here "
                f"(onset over {weather.dew_onset_over}). Consider pace."
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

    # Bowler summary
    bowler_summary: dict[str, list[int]] = {}
    for over, b in assignments.items():
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

    # Death pair — explain selection based on wickets + economy
    if death_pool:
        d1 = death_pool[0]
        d2 = death_pool[1] if len(death_pool) > 1 else None
        s1 = stats.get(d1, {})
        fit1 = _phase_fitness(d1, "Death", stats, meta, weather)
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

    # Spinner allocation — weather driven
    spin_bowlers = [b for b in our_bowlers if _bowl_type(b, meta) == "spin"]
    if spin_bowlers:
        if weather.severe_dew:
            key_decisions.append(
                f"SEVERE DEW — {', '.join(spin_bowlers)} restricted to overs 7-14 only. "
                f"No spinners after over {weather.dew_onset_over}."
            )
        elif weather.spinner_penalty <= 0.75:
            key_decisions.append(
                f"Moderate dew (penalty {weather.spinner_penalty:.2f}) — keep "
                f"{', '.join(spin_bowlers)} in middle overs (7-14), avoid death."
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
        tag         = " (estimated)" if estimated else ""

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

    key_decisions = key_decisions[:7]  # allow up to 7 with opp intelligence

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

    # Validate cap compliance — the plan generator must never produce a plan
    # where any bowler is assigned more than 4 overs. If this fires, it's a
    # bug in the allocation logic above, not a live-match deviation.
    for bowler, alloc in bowler_summary.items():
        if len(alloc) > MAX_OVERS_PER_BOWLER:
            raise RuntimeError(
                f"[bowling_plan] BUG: {bowler} allocated {len(alloc)} overs "
                f"(cap = {MAX_OVERS_PER_BOWLER}). Fix the allocation logic."
            )

    return BowlingPlan(
        overs          = over_assignments,
        bowler_summary = bowler_summary,
        key_decisions  = key_decisions,
        contingencies  = contingencies,
    )


# ---------------------------------------------------------------------------
# QUICK SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from utils.situation import WeatherImpact

    lahore_bowlers = [
        "Shaheen Shah Afridi",
        "Haris Rauf",
        "Zaman Khan",
        "Rashid Khan",
        "Liam Dawson",
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

    plan = generate_bowling_plan(
        our_bowlers     = lahore_bowlers,
        weather         = weather_dew,
        venue           = "Gaddafi Stadium, Lahore",
        opposition_team = "Karachi Kings",
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
    print()
