# pipeline/build_opposition_profiles.py
# Builds opposition_profiles.csv — one row per team per season.
# Used by engine/opposition_predictor.py to predict batting orders
# and bowling plans for the opposition.
#
# Run directly:  python pipeline/build_opposition_profiles.py
# Or import:     from pipeline.build_opposition_profiles import run

from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

PROJ_ROOT     = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJ_ROOT / "data" / "processed"
BBB_FILE      = PROCESSED_DIR / "ball_by_ball.parquet"
PLAYER_INDEX  = PROCESSED_DIR / "player_index.csv"
SCHEDULE_SPEC = PROJ_ROOT / "data" / "psl_schedule.csv"
SCHEDULE_ACT  = PROJ_ROOT.parent / "psl_schedule.csv"
OUTPUT_FILE   = PROCESSED_DIR / "opposition_profiles.csv"

# ---------------------------------------------------------------------------
# PLAYER STYLE LOOKUPS  (loaded once from player_index.csv)
# ---------------------------------------------------------------------------

def _load_player_styles(path: Path) -> tuple[dict, dict, dict, dict]:
    """
    Returns:
        batting_hand   {name: 'Left-hand bat' | 'Right-hand bat'}
        bowl_type      {name: 'pace' | 'spin' | 'unknown' | 'non-bowler'}
        bowl_arm       {name: 'left' | 'right' | 'unknown'}
        spin_subtype   {name: 'legspin' | 'offspin' | 'leftarm_spin'}
                       (only populated for spin bowlers with a known subtype)
    """
    batting_hand: dict[str, str] = {}
    bowl_type:    dict[str, str] = {}
    bowl_arm:     dict[str, str] = {}
    spin_subtype: dict[str, str] = {}

    if not path.exists():
        return batting_hand, bowl_type, bowl_arm, spin_subtype

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name  = row.get("player_name", "").strip()
            bat   = (row.get("batting_style") or "").strip()
            bowl  = (row.get("bowling_style") or "").lower().strip()
            role  = (row.get("primary_role")  or "").lower().strip()
            if not name:
                continue
            batting_hand[name] = bat

            if any(w in bowl for w in ("fast", "medium")):
                bowl_type[name] = "pace"
            elif any(w in bowl for w in ("break", "orthodox", "spin", "leg")):
                bowl_type[name] = "spin"
            elif not bowl and any(w in role for w in ("batsman", "wk")):
                bowl_type[name] = "non-bowler"
            else:
                bowl_type[name] = "unknown"

            bowl_arm[name] = "left" if bowl.startswith("left") else (
                "right" if bowl.startswith("right") else "unknown"
            )

            # Spin subtype — only set for confirmed spin bowlers
            if bowl_type[name] == "spin":
                if any(w in bowl for w in ("leg-break", "wrist-spin")):
                    spin_subtype[name] = "legspin"
                elif "off-break" in bowl:
                    spin_subtype[name] = "offspin"
                elif any(w in bowl for w in ("orthodox", "slow left-arm")):
                    spin_subtype[name] = "leftarm_spin"

    return batting_hand, bowl_type, bowl_arm, spin_subtype


# ---------------------------------------------------------------------------
# BATTING POSITION EXTRACTOR
# ---------------------------------------------------------------------------

def _extract_batting_positions(bbb: pd.DataFrame) -> pd.DataFrame:
    """
    Derive batting position (1-11) for every player in every innings.
    Method: chronological first appearance as batter OR non_striker.
    Position 1 = faces first ball; Position 2 = their opening partner.
    Returns DataFrame: match_id, season, batting_team, batter, batting_position
    """
    records = []

    for (match_id, innings_num), grp in bbb.groupby(["match_id", "innings"]):
        grp_sorted = grp.sort_values(["over", "ball"])
        batting_team = grp_sorted["batting_team"].iloc[0]
        season       = int(grp_sorted["season"].iloc[0])

        seen: dict[str, int] = {}
        pos = 1

        for _, row in grp_sorted.iterrows():
            for player in (row["batter"], row["non_striker"]):
                if player and player not in seen:
                    seen[player] = pos
                    pos += 1

        for player, position in seen.items():
            records.append({
                "match_id":        match_id,
                "season":          season,
                "batting_team":    batting_team,
                "batter":          player,
                "batting_position": position,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# TYPICAL BATTING ORDER
# ---------------------------------------------------------------------------

def _typical_batting_order(
    pos_df: pd.DataFrame,
    team: str,
    season: int,
) -> dict[str, str]:
    """
    For a given team+season, return {position_str: player_name} for positions 1-8.
    Uses most-frequent player at each position weighted by recency.
    season=0 means aggregate across all seasons (with recency weighting).
    """
    if season == 0:
        subset = pos_df[pos_df["batting_team"] == team].copy()
        # Weight recent seasons more: 2016=1, 2017=2, ..., 2025=10
        season_min = int(subset["season"].min()) if len(subset) else 2016
        subset["weight"] = subset["season"] - season_min + 1
    else:
        subset = pos_df[
            (pos_df["batting_team"] == team) &
            (pos_df["season"] == season)
        ].copy()
        subset["weight"] = 1

    if subset.empty:
        return {}

    order = {}
    assigned: set[str] = set()

    for pos in range(1, 9):
        pos_players = subset[subset["batting_position"] == pos]
        if pos_players.empty:
            continue
        # Weighted vote: player with most appearances at this position
        scores = pos_players.groupby("batter")["weight"].sum().sort_values(ascending=False)
        # Pick highest-scoring player not already assigned to an earlier position
        for player, _ in scores.items():
            if player not in assigned:
                order[str(pos)] = player
                assigned.add(player)
                break

    return order


# ---------------------------------------------------------------------------
# BOWLING ORDER
# ---------------------------------------------------------------------------

def _typical_bowling_order(
    bbb: pd.DataFrame,
    team: str,
    season: int,
) -> dict[str, str]:
    """
    For a given team+season, return {over_str: bowler_name} for overs 1-20.
    season=0 = all-season aggregate with recency weighting.
    """
    if season == 0:
        subset = bbb[bbb["bowling_team"] == team].copy()
        season_min = int(subset["season"].min()) if len(subset) else 2016
        subset["weight"] = subset["season"] - season_min + 1
    else:
        subset = bbb[
            (bbb["bowling_team"] == team) &
            (bbb["season"] == season)
        ].copy()
        subset["weight"] = 1

    if subset.empty:
        return {}

    # For each over 0-19, find the bowler who bowled there most often
    # (taking the modal bowler per match for each over, then aggregating)
    over_bowler_map: dict[int, Counter] = defaultdict(Counter)

    for (match_id,), mdf in subset.groupby(["match_id"]):
        season_val = int(mdf["season"].iloc[0])
        w = season_val - (int(bbb["season"].min()) or 2016) + 1 if season == 0 else 1
        for over_num, odf in mdf.groupby("over"):
            bowler_counts = odf["bowler"].value_counts()
            top_bowler = bowler_counts.idxmax() if len(bowler_counts) else None
            if top_bowler:
                over_bowler_map[over_num][top_bowler] += w

    result = {}
    for over_num in range(20):
        if over_num in over_bowler_map:
            top = over_bowler_map[over_num].most_common(1)
            if top:
                result[str(over_num + 1)] = top[0][0]   # 1-indexed for display

    return result


# ---------------------------------------------------------------------------
# PHASE STRIKE RATE
# ---------------------------------------------------------------------------

def _phase_sr(batting_df: pd.DataFrame, phase: str) -> float:
    sub = batting_df[batting_df["phase"] == phase & ~batting_df["is_wide"]] if isinstance(phase, str) else batting_df[~batting_df["is_wide"]]
    balls = len(sub)
    runs  = sub["runs_batter"].sum()
    return round(runs * 100 / balls, 1) if balls > 0 else 0.0


# ---------------------------------------------------------------------------
# AGGRESSIVE OPENER %
# ---------------------------------------------------------------------------

def _aggressive_opener_pct(
    bbb: pd.DataFrame,
    pos_df: pd.DataFrame,
    team: str,
    season: int,
) -> float:
    """% of innings where the team's opener scored at SR > 130 in powerplay."""
    if season == 0:
        bat = bbb[(bbb["batting_team"] == team) & (bbb["phase"] == "powerplay") & (~bbb["is_wide"])]
        pd_sub = pos_df[(pos_df["batting_team"] == team) & (pos_df["batting_position"] == 1)]
    else:
        bat = bbb[
            (bbb["batting_team"] == team) &
            (bbb["season"] == season) &
            (bbb["phase"] == "powerplay") &
            (~bbb["is_wide"])
        ]
        pd_sub = pos_df[
            (pos_df["batting_team"] == team) &
            (pos_df["season"] == season) &
            (pos_df["batting_position"] == 1)
        ]

    if bat.empty or pd_sub.empty:
        return 0.0

    openers_by_match = dict(zip(pd_sub["match_id"], pd_sub["batter"]))
    aggressive = 0
    total = 0

    for match_id, opener in openers_by_match.items():
        opener_balls = bat[(bat["match_id"] == match_id) & (bat["batter"] == opener)]
        if opener_balls.empty:
            continue
        balls = len(opener_balls)
        runs  = opener_balls["runs_batter"].sum()
        sr    = runs * 100 / balls if balls > 0 else 0
        if sr > 130:
            aggressive += 1
        total += 1

    return round(aggressive * 100 / total, 1) if total > 0 else 0.0


# ---------------------------------------------------------------------------
# VS BOWLING TYPE ECONOMY
# ---------------------------------------------------------------------------

def _vs_type_economy(
    bbb: pd.DataFrame,
    bowl_type: dict,
    bowl_arm: dict,
    team: str,
    season: int,
    type_filter: str,      # 'pace' | 'spin'
    arm_filter: str = "",  # 'left' | '' = any
) -> float:
    """Runs per over a team scores when batting against a specific bowling type."""
    if season == 0:
        sub = bbb[(bbb["batting_team"] == team) & (~bbb["is_wide"])].copy()
    else:
        sub = bbb[
            (bbb["batting_team"] == team) &
            (bbb["season"] == season) &
            (~bbb["is_wide"])
        ].copy()

    sub["bowl_type"] = sub["bowler"].map(lambda b: bowl_type.get(b, "unknown"))
    sub["bowl_arm"]  = sub["bowler"].map(lambda b: bowl_arm.get(b, "unknown"))

    filtered = sub[sub["bowl_type"] == type_filter]
    if arm_filter:
        filtered = filtered[filtered["bowl_arm"] == arm_filter]

    balls = len(filtered)
    runs  = filtered["runs_batter"].sum() + filtered["runs_extras"].sum()
    return round(runs * 6 / balls, 2) if balls > 0 else 0.0


# ---------------------------------------------------------------------------
# VS SPIN SUBTYPE ECONOMY
# ---------------------------------------------------------------------------

def _vs_spintype_economy(
    bbb: pd.DataFrame,
    bowl_type: dict,
    spin_subtype: dict,
    team: str,
    season: int,
    subtype: str,   # 'legspin' | 'offspin' | 'leftarm_spin'
) -> float:
    """Runs per over a team scores when batting against a specific spin subtype."""
    if season == 0:
        sub = bbb[(bbb["batting_team"] == team) & (~bbb["is_wide"])].copy()
    else:
        sub = bbb[
            (bbb["batting_team"] == team) &
            (bbb["season"] == season) &
            (~bbb["is_wide"])
        ].copy()

    sub["_bowl_type"]    = sub["bowler"].map(lambda b: bowl_type.get(b, "unknown"))
    sub["_spin_subtype"] = sub["bowler"].map(lambda b: spin_subtype.get(b, ""))

    filtered = sub[
        (sub["_bowl_type"] == "spin") &
        (sub["_spin_subtype"] == subtype)
    ]

    balls = len(filtered)
    runs  = filtered["runs_batter"].sum() + filtered["runs_extras"].sum()
    return round(runs * 6 / balls, 2) if balls > 0 else 0.0


# ---------------------------------------------------------------------------
# CHASE / DEFEND WIN PCT
# ---------------------------------------------------------------------------

def _load_chase_stats(
    bbb: pd.DataFrame,
    schedule_path: Optional[Path],
) -> pd.DataFrame:
    """
    Returns DataFrame: team, season, chase_wins, defend_wins, total_matches
    Uses schedule for winner; falls back to ball_by_ball derivation.
    """
    # Derive from ball_by_ball: innings 2 final score vs target
    inn2 = (
        bbb[bbb["innings"] == 2]
        .sort_values(["match_id", "over", "ball"])
        .groupby("match_id")
        .agg(
            final_score  =("innings_score", "last"),
            target       =("target", "last"),
            batting_team2=("batting_team", "first"),
            bowling_team2=("bowling_team", "first"),
            season       =("season", "first"),
        )
        .reset_index()
    )
    inn2["chased"] = inn2["final_score"] >= inn2["target"]

    records = []
    for _, row in inn2.iterrows():
        s = int(row["season"])
        # Team that batted second (the chaser)
        records.append({
            "team": row["batting_team2"], "season": s,
            "chase_win": int(row["chased"]), "defend_win": 0,
        })
        # Team that batted first (the defender)
        records.append({
            "team": row["bowling_team2"], "season": s,
            "chase_win": 0, "defend_win": int(not row["chased"]),
        })

    chase_df = pd.DataFrame(records)
    agg = chase_df.groupby(["team", "season"]).agg(
        chase_wins  =("chase_win",  "sum"),
        defend_wins =("defend_win", "sum"),
    ).reset_index()
    agg["total"] = agg["chase_wins"] + agg["defend_wins"]
    agg["chase_win_pct"]  = (agg["chase_wins"]  * 100 / agg["total"]).round(1)
    agg["defend_win_pct"] = (agg["defend_wins"] * 100 / agg["total"]).round(1)

    # Career row (season=0)
    career = agg.groupby("team").agg(
        chase_wins  =("chase_wins",  "sum"),
        defend_wins =("defend_wins", "sum"),
        total       =("total",       "sum"),
    ).reset_index()
    career["season"]        = 0
    career["chase_win_pct"] = (career["chase_wins"] * 100 / career["total"]).round(1)
    career["defend_win_pct"]= (career["defend_wins"]* 100 / career["total"]).round(1)

    return pd.concat([agg, career], ignore_index=True)


# ---------------------------------------------------------------------------
# MAIN BUILD FUNCTION
# ---------------------------------------------------------------------------

def run(
    bbb_path:    Optional[Path] = None,
    output_path: Optional[Path] = None,
    verbose:     bool = True,
) -> pd.DataFrame:
    t0 = time.time()

    bbb_path    = Path(bbb_path    or BBB_FILE)
    output_path = Path(output_path or OUTPUT_FILE)

    if not bbb_path.exists():
        raise FileNotFoundError(f"ball_by_ball.parquet not found at {bbb_path}. Run parse_cricsheet.py first.")

    if verbose:
        print(f"[build_opposition_profiles] Loading {bbb_path.name}...")
    bbb = pd.read_parquet(bbb_path)

    # Load player style lookups
    batting_hand, bowl_type, bowl_arm, spin_subtype = _load_player_styles(PLAYER_INDEX)
    if verbose:
        print(f"[build_opposition_profiles] Player styles loaded: {len(batting_hand)} players")

    # Extract batting positions for all matches
    if verbose:
        print("[build_opposition_profiles] Extracting batting positions...")
    pos_df = _extract_batting_positions(bbb)

    # Chase/defend stats
    sched_path = SCHEDULE_SPEC if SCHEDULE_SPEC.exists() else (SCHEDULE_ACT if SCHEDULE_ACT.exists() else None)
    chase_df = _load_chase_stats(bbb, sched_path)

    teams   = sorted(bbb["batting_team"].unique())
    seasons = sorted(bbb["season"].unique())
    # Also add season=0 (all-time aggregate)
    seasons_with_career = [0] + list(seasons)

    rows = []

    for team in teams:
        if verbose:
            print(f"  Processing {team}...")

        for season in seasons_with_career:
            if season == 0:
                bat = bbb[(bbb["batting_team"] == team)].copy()
                bow = bbb[(bbb["bowling_team"] == team)].copy()
            else:
                bat = bbb[(bbb["batting_team"] == team) & (bbb["season"] == season)].copy()
                bow = bbb[(bbb["bowling_team"] == team) & (bbb["season"] == season)].copy()

            # Skip seasons where the team has no data
            if bat.empty:
                continue

            # ── Batting order ─────────────────────────────────────────────
            batting_order = _typical_batting_order(pos_df, team, season)

            # ── Left-hand top-6 pct ───────────────────────────────────────
            top6_batters = [
                batting_order.get(str(p)) for p in range(1, 7)
                if batting_order.get(str(p))
            ]
            lh_batters = {p for p in top6_batters if batting_hand.get(p, "").startswith("Left")}
            left_hand_top6_pct = round(len(lh_batters) * 100 / max(len(top6_batters), 1), 1)

            # ── Aggressive opener pct ─────────────────────────────────────
            agg_opener_pct = _aggressive_opener_pct(bbb, pos_df, team, season)

            # ── Phase strike rates (batting) ──────────────────────────────
            legal_bat = bat[~bat["is_wide"]]

            def _sr(phase_label):
                sub = legal_bat[legal_bat["phase"] == phase_label]
                b, r = len(sub), sub["runs_batter"].sum()
                return round(r * 100 / b, 1) if b > 0 else 0.0

            pp_sr  = _sr("powerplay")
            mid_sr = _sr("middle")
            dth_sr = _sr("death")

            # ── vs bowling type economies (batting) ───────────────────────
            vs_spin = _vs_type_economy(bbb, bowl_type, bowl_arm, team, season, "spin")
            vs_pace = _vs_type_economy(bbb, bowl_type, bowl_arm, team, season, "pace")
            vs_left = _vs_type_economy(bbb, bowl_type, bowl_arm, team, season, "pace", arm_filter="left")
            if vs_left == 0.0:
                vs_left = _vs_type_economy(bbb, bowl_type, bowl_arm, team, season, "spin", arm_filter="left")

            # ── vs spin subtype economies ──────────────────────────────────
            vs_legspin    = _vs_spintype_economy(bbb, bowl_type, spin_subtype, team, season, "legspin")
            vs_offspin    = _vs_spintype_economy(bbb, bowl_type, spin_subtype, team, season, "offspin")
            vs_leftarm_sp = _vs_spintype_economy(bbb, bowl_type, spin_subtype, team, season, "leftarm_spin")

            # ── Chase / defend win pct ────────────────────────────────────
            chase_row = chase_df[(chase_df["team"] == team) & (chase_df["season"] == season)]
            if not chase_row.empty:
                chase_win_pct  = float(chase_row.iloc[0]["chase_win_pct"])
                defend_win_pct = float(chase_row.iloc[0]["defend_win_pct"])
            else:
                chase_win_pct = defend_win_pct = 0.0

            # ── Bowling order ─────────────────────────────────────────────
            bowling_order = _typical_bowling_order(bbb, team, season)

            # ── Bowling economies ─────────────────────────────────────────
            legal_bow = bow[~bow["is_wide"]]

            def _eco(phase_label):
                sub = legal_bow[legal_bow["phase"] == phase_label]
                b, r = len(sub), sub["runs_total"].sum()
                return round(r * 6 / b, 2) if b > 0 else 0.0

            pp_bowl_eco  = _eco("powerplay")
            dth_bowl_eco = _eco("death")

            # ── Pace / spin overs pct ─────────────────────────────────────
            legal_bow["bowl_type"] = legal_bow["bowler"].map(lambda b: bowl_type.get(b, "unknown"))
            total_legal = len(legal_bow)
            pace_balls  = (legal_bow["bowl_type"] == "pace").sum()
            spin_balls  = (legal_bow["bowl_type"] == "spin").sum()
            pace_overs_pct = round(pace_balls * 100 / total_legal, 1) if total_legal > 0 else 0.0
            spin_overs_pct = round(spin_balls * 100 / total_legal, 1) if total_legal > 0 else 0.0

            rows.append({
                "team":                     team,
                "season":                   season,
                "typical_batting_order":    json.dumps(batting_order),
                "left_hand_top6_pct":       left_hand_top6_pct,
                "aggressive_opener_pct":    agg_opener_pct,
                "powerplay_sr":             pp_sr,
                "middle_sr":                mid_sr,
                "death_sr":                 dth_sr,
                "vs_spin_economy":          vs_spin,
                "vs_pace_economy":          vs_pace,
                "vs_leftarm_economy":       vs_left,
                "vs_legspin_economy":       vs_legspin,
                "vs_offspin_economy":       vs_offspin,
                "vs_leftarm_spin_economy":  vs_leftarm_sp,
                "chase_win_pct":            chase_win_pct,
                "defend_win_pct":           defend_win_pct,
                "likely_bowling_order":     json.dumps(bowling_order),
                "powerplay_bowling_economy":pp_bowl_eco,
                "death_bowling_economy":    dth_bowl_eco,
                "pace_overs_pct":           pace_overs_pct,
                "spin_overs_pct":           spin_overs_pct,
            })

    df_out = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)

    elapsed = time.time() - t0
    if verbose:
        print(f"[build_opposition_profiles] {len(df_out)} rows written -> {output_path.name}  [{elapsed:.1f}s]")

    return df_out


# ---------------------------------------------------------------------------
# SUMMARY + ENTRY POINT
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print("  opposition_profiles.csv -- summary")
    print(f"{'='*60}")
    print(f"  Rows    : {len(df)}")
    print(f"  Teams   : {sorted(df['team'].unique())}")
    print(f"  Seasons : {sorted(df['season'].unique())}")
    print()

    # Show all-time profiles (season=0) for each team
    career = df[df["season"] == 0].sort_values("team")
    print(f"  {'Team':<28} PP_SR  Mid_SR  Dth_SR  Chase%  Defend%  Pace%  Spin%")
    print(f"  {'-'*80}")
    for _, r in career.iterrows():
        order = json.loads(r["typical_batting_order"])
        top3  = " / ".join(order.get(str(p), "?") for p in range(1, 4))
        print(f"  {r['team']:<28} {r['powerplay_sr']:>5.1f}  {r['middle_sr']:>5.1f}  "
              f"{r['death_sr']:>5.1f}  {r['chase_win_pct']:>5.1f}  {r['defend_win_pct']:>6.1f}  "
              f"{r['pace_overs_pct']:>4.1f}  {r['spin_overs_pct']:>4.1f}")
        print(f"    Typical top-3: {top3}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    df = run(verbose=True)
    _print_summary(df)
