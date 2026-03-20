# pipeline/build_partnership_history.py
# Extracts all batting partnerships from ball_by_ball.parquet
# -> produces data/processed/partnership_history.parquet
#
# Schema (one row per batter pair per season, plus season=0 career):
#   batter1, batter2             -- alphabetically ordered pair
#   season                       -- 0 = career aggregate
#   occurrences                  -- how many times they batted together
#   avg_runs, avg_balls          -- average partnership length
#   avg_sr                       -- average strike rate
#   broken_by_pace_pct           -- % times ended by a pace bowler wicket
#   broken_by_spin_pct           -- % times ended by a spin bowler wicket
#   broken_by_bowling_change_pct -- % times ended within 2 balls of a bowling change
#   avg_over_when_broken         -- average over number when partnership ended
#   max_partnership_runs         -- highest single partnership
#
# Run directly: python pipeline/build_partnership_history.py
# Or import:    from pipeline.build_partnership_history import run

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

PROJ_ROOT      = Path(__file__).resolve().parent.parent
PROCESSED_DIR  = PROJ_ROOT / "data" / "processed"
BBB_PATH       = PROCESSED_DIR / "ball_by_ball.parquet"
OUTPUT_FILE    = PROCESSED_DIR / "partnership_history.parquet"
PLAYER_INDEX   = PROCESSED_DIR / "player_index.csv"

# Fallback player index
PLAYER_INDEX_FALLBACK = PROJ_ROOT.parent / "player_index.csv"


# ---------------------------------------------------------------------------
# PLAYER BOWL TYPE LOOKUP  (pace vs spin, from player_index.csv)
# ---------------------------------------------------------------------------

def _load_bowl_types(player_index_path: Path) -> dict[str, str]:
    """Returns {player_name: 'pace'|'spin'|'unknown'}."""
    bowl_types: dict[str, str] = {}
    if not player_index_path.exists():
        return bowl_types

    with open(player_index_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name  = row.get("player_name", "").strip()
            style = (row.get("bowling_style") or "").lower().strip()
            role  = (row.get("primary_role")  or "").lower()
            if not name:
                continue
            if not style and any(w in role for w in ("batsman", "wk")):
                bowl_types[name] = "non-bowler"
                continue
            if any(w in style for w in ("fast", "medium", "pace", "seam", "swing")):
                bowl_types[name] = "pace"
            elif any(w in style for w in ("spin", "off", "leg", "googly", "chinaman", "slow")):
                bowl_types[name] = "spin"
            else:
                bowl_types[name] = "unknown"

    return bowl_types


# ---------------------------------------------------------------------------
# PARTNERSHIP EXTRACTION FROM A SINGLE INNINGS
# ---------------------------------------------------------------------------

def _extract_innings_partnerships(
    group: pd.DataFrame,
    bowl_types: dict[str, str],
) -> list[dict]:
    """
    Extract individual partnership records from one match-innings group.
    Returns a list of dicts, one per partnership occurrence.
    """
    records: list[dict] = []

    # Sort deliveries in order
    grp = group.sort_values(["over", "ball"]).reset_index(drop=True)

    if grp.empty:
        return records

    meta = grp.iloc[0]
    match_id = meta["match_id"]
    season   = meta["season"]
    venue    = meta["venue"]
    innings  = meta["innings"]

    current_pair: tuple[str, str] | None = None
    p_runs   = 0
    p_balls  = 0
    p_start_over = 0
    bowler_history: list[str] = []   # recent bowlers for bowling-change detection

    def _pair(b: str, ns: str) -> tuple[str, str]:
        return tuple(sorted([b, ns]))  # type: ignore[return-value]

    def _was_bowling_change(bowler_history: list[str]) -> bool:
        """True if the most recent bowler appeared within last 2 legal deliveries
        after a different bowler (i.e., bowling change occurred very recently)."""
        if len(bowler_history) < 3:
            return False
        # Look back: if bowler at [-1] is different from [-2] or [-3]
        recent = bowler_history[-3:]
        return recent[-1] != recent[-2]

    def _save(p_runs, p_balls, pair, bowler, wkt_type, broken_over):
        if p_balls < 1:
            return
        b_type    = bowl_types.get(bowler, "unknown") if bowler else "none"
        is_pace   = int(b_type == "pace"  and wkt_type not in ("run out", ""))
        is_spin   = int(b_type == "spin"  and wkt_type not in ("run out", ""))
        is_change = int(_was_bowling_change(bowler_history))
        records.append({
            "batter1":           pair[0],
            "batter2":           pair[1],
            "season":            season,
            "match_id":          match_id,
            "venue":             venue,
            "innings":           innings,
            "partnership_runs":  p_runs,
            "partnership_balls": p_balls,
            "partnership_sr":    round(p_runs / p_balls * 100, 2) if p_balls else 0.0,
            "broken_by_pace":    is_pace,
            "broken_by_spin":    is_spin,
            "bowling_change":    is_change,
            "over_ended":        broken_over,
            "was_broken":        int(bool(wkt_type)),   # 0 = end of innings/not out
        })

    for _, row in grp.iterrows():
        batter      = str(row["batter"])
        non_striker = str(row["non_striker"])
        bowler      = str(row["bowler"])
        is_wide     = bool(row["is_wide"])
        is_wicket   = bool(row["is_wicket"])
        wkt_type    = str(row["wicket_type"])
        over        = int(row["over"])
        runs_total  = int(row["runs_total"])

        new_pair = _pair(batter, non_striker)

        # Detect pair change (wicket already fell on previous delivery and new batter is in)
        if current_pair is None:
            current_pair  = new_pair
            p_start_over  = over
            p_runs        = 0
            p_balls       = 0
        elif new_pair != current_pair:
            # Partnership ended — save old one (wicket was on previous ball, no wkt_type here)
            _save(p_runs, p_balls, current_pair, bowler, "", over)
            current_pair  = new_pair
            p_start_over  = over
            p_runs        = 0
            p_balls       = 0

        # Accumulate
        p_runs += runs_total
        if not is_wide:
            p_balls += 1

        # Track bowler history (non-wide deliveries for bowling-change detection)
        if not is_wide:
            bowler_history.append(bowler)
            if len(bowler_history) > 6:
                bowler_history.pop(0)

        # If wicket on this ball, save the partnership (pair will differ next delivery)
        if is_wicket:
            _save(p_runs, p_balls, current_pair, bowler, wkt_type, over)
            current_pair  = None   # force reset on next ball
            p_runs        = 0
            p_balls       = 0
            bowler_history.clear()

    # Save final not-out partnership
    if current_pair is not None and p_balls > 0:
        _save(p_runs, p_balls, current_pair, "", "", p_start_over)

    return records


# ---------------------------------------------------------------------------
# AGGREGATE RAW OCCURRENCES -> SUMMARY ROWS
# ---------------------------------------------------------------------------

def _aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse raw occurrence rows into one row per (batter1, batter2, season).
    Also adds season=0 career aggregates.
    """
    def _agg(grp: pd.DataFrame) -> pd.Series:
        n = len(grp)
        broken = grp[grp["was_broken"] == 1]
        n_broken = len(broken)
        return pd.Series({
            "occurrences":                n,
            "avg_runs":                   round(grp["partnership_runs"].mean(),  1),
            "avg_balls":                  round(grp["partnership_balls"].mean(), 1),
            "avg_sr":                     round(grp["partnership_sr"].mean(),    1),
            "broken_by_pace_pct":         round(broken["broken_by_pace"].sum() / n_broken * 100, 1) if n_broken else 0.0,
            "broken_by_spin_pct":         round(broken["broken_by_spin"].sum() / n_broken * 100, 1) if n_broken else 0.0,
            "broken_by_bowling_change_pct": round(broken["bowling_change"].sum() / n_broken * 100, 1) if n_broken else 0.0,
            "avg_over_when_broken":       round(broken["over_ended"].mean(), 1) if n_broken else 0.0,
            "max_partnership_runs":       int(grp["partnership_runs"].max()),
        })

    seasonal = (
        raw
        .groupby(["batter1", "batter2", "season"], observed=True)
        .apply(_agg)
        .reset_index()
    )

    # Career (season=0)
    raw_career = raw.copy()
    raw_career["season"] = 0
    career = (
        raw_career
        .groupby(["batter1", "batter2", "season"], observed=True)
        .apply(_agg)
        .reset_index()
    )

    combined = pd.concat([seasonal, career], ignore_index=True)
    combined["season"] = combined["season"].astype("int16")
    return combined.sort_values(["batter1", "batter2", "season"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------------------

def run(
    bbb_path:    Optional[Path] = None,
    output_path: Optional[Path] = None,
    verbose:     bool = True,
) -> pd.DataFrame:
    """
    Build partnership_history.parquet from ball_by_ball.parquet.

    Returns:
        DataFrame with one row per (batter1, batter2, season).
    """
    t0 = time.time()

    bbb_path = Path(bbb_path) if bbb_path else BBB_PATH
    output_path = Path(output_path) if output_path else OUTPUT_FILE

    if verbose:
        print("[build_partnership_history] Loading ball_by_ball.parquet...")
    bbb = pd.read_parquet(bbb_path)

    # Load bowl types
    pi_path = PLAYER_INDEX if PLAYER_INDEX.exists() else PLAYER_INDEX_FALLBACK
    bowl_types = _load_bowl_types(pi_path) if pi_path.exists() else {}
    if verbose:
        print(f"[build_partnership_history] Bowl types loaded: {len(bowl_types)} players")

    # Extract all partnerships
    if verbose:
        print("[build_partnership_history] Extracting partnerships...")

    all_records: list[dict] = []
    groups = bbb.groupby(["match_id", "innings"], observed=True)
    n_groups = len(groups)

    for i, ((mid, inn), grp) in enumerate(groups, 1):
        records = _extract_innings_partnerships(grp, bowl_types)
        all_records.extend(records)
        if verbose and i % 100 == 0:
            print(f"  {i}/{n_groups} innings processed...")

    if verbose:
        print(f"[build_partnership_history] {len(all_records):,} raw partnership occurrences extracted")

    raw_df = pd.DataFrame(all_records)

    # Aggregate
    if verbose:
        print("[build_partnership_history] Aggregating by pair and season...")
    summary = _aggregate(raw_df)

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(output_path, index=False)

    elapsed = time.time() - t0
    if verbose:
        print(
            f"[build_partnership_history] {len(summary):,} rows written -> "
            f"partnership_history.parquet  [{elapsed:.1f}s]"
        )

    return summary


# ---------------------------------------------------------------------------
# SUMMARY PRINTER
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame) -> None:
    career = df[df["season"] == 0]
    seasonal = df[df["season"] > 0]

    print(f"\n{'='*60}")
    print(f"  partnership_history.parquet -- summary")
    print(f"{'='*60}")
    print(f"  Total rows       : {len(df):,}")
    print(f"  Unique pairs     : {len(career):,}")
    print(f"  Seasonal rows    : {len(seasonal):,}")
    print(f"  Seasons          : {sorted(df['season'].unique().tolist())}")

    # Top partnerships by avg runs (career, min 3 occurrences)
    top = (
        career[career["occurrences"] >= 3]
        .sort_values("avg_runs", ascending=False)
        .head(10)
    )
    print(f"\n  Top 10 partnerships by avg runs (min 3 occurrences):")
    print(f"  {'Pair':<40}  Occ  AvgR  AvgB   SR    PaceBrk%")
    print(f"  {'-'*70}")
    for _, r in top.iterrows():
        pair = f"{r['batter1']} / {r['batter2']}"
        print(
            f"  {pair:<40}  {int(r['occurrences']):>3}  "
            f"{r['avg_runs']:>5.1f}  {r['avg_balls']:>4.1f}  "
            f"{r['avg_sr']:>5.1f}  {r['broken_by_pace_pct']:>5.1f}%"
        )

    print(f"\n  Pairs broken most by spin:")
    top_spin = (
        career[career["occurrences"] >= 3]
        .sort_values("broken_by_spin_pct", ascending=False)
        .head(5)
    )
    for _, r in top_spin.iterrows():
        print(
            f"    {r['batter1']} / {r['batter2']}  "
            f"spin_brk={r['broken_by_spin_pct']:.0f}%  ({int(r['occurrences'])} occ)"
        )

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = run(verbose=True)
    _print_summary(df)
