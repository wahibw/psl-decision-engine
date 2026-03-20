# pipeline/parse_cricsheet.py
# Reads every Cricsheet PSL JSON file -> produces data/processed/ball_by_ball.parquet
#
# Schema (one row per delivery):
#   match_id, season, date, venue, batting_team, bowling_team
#   innings, over, ball, legal_ball
#   batter, bowler, non_striker
#   runs_batter, runs_extras, runs_total
#   is_wide, is_noball, is_wicket, wicket_type
#   phase, innings_score, innings_wickets
#   target, required_runs, balls_remaining, rrr, crr
#
# Run directly:  python pipeline/parse_cricsheet.py
# Or import:     from pipeline.parse_cricsheet import run

from __future__ import annotations

import csv
import glob
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# PATHS  - resolved relative to this file so the script works from any cwd
# ---------------------------------------------------------------------------

PROJ_ROOT     = Path(__file__).resolve().parent.parent
RAW_DIR       = PROJ_ROOT / "data" / "raw"
PROCESSED_DIR = PROJ_ROOT / "data" / "processed"
OUTPUT_FILE   = PROCESSED_DIR / "ball_by_ball.parquet"

# Fallback raw data location (where we actually put the files)
RAW_FALLBACK  = PROJ_ROOT.parent / "psl_json"

# player_index.csv locations (spec vs. actual)
PLAYER_INDEX_SPEC   = PROCESSED_DIR / "player_index.csv"
PLAYER_INDEX_ACTUAL = PROJ_ROOT.parent / "player_index.csv"

# ---------------------------------------------------------------------------
# VENUE NAME NORMALISATION  (same as psl_schedule.csv)
# ---------------------------------------------------------------------------

VENUE_FIX = {
    "National Stadium":       "National Stadium, Karachi",
    "Gaddafi Stadium":        "Gaddafi Stadium, Lahore",
    "Sheikh Zayed Stadium":   "Sheikh Zayed Stadium, Abu Dhabi",
}

# ---------------------------------------------------------------------------
# PHASE LABELS
# ---------------------------------------------------------------------------

def _phase(over: int) -> str:
    if over < 6:
        return "powerplay"
    elif over < 15:
        return "middle"
    elif over < 20:
        return "death"
    return "super_over"


# ---------------------------------------------------------------------------
# ALIAS MAP  (built from player_index.csv name_variants column)
# ---------------------------------------------------------------------------

def _load_alias_map(player_index_path: Path) -> dict[str, str]:
    """
    Returns {alias_name: canonical_name}.
    Only covers confirmed variants - unknown names are kept as-is.
    """
    alias: dict[str, str] = {}
    if not player_index_path.exists():
        return alias
    with open(player_index_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            variants = (row.get("name_variants") or "").strip()
            if variants:
                canonical = row["player_name"].strip()
                for v in variants.split(";"):
                    v = v.strip()
                    if v:
                        alias[v] = canonical
    return alias


def _resolve(name: str, alias: dict[str, str]) -> str:
    return alias.get(name, name)


# ---------------------------------------------------------------------------
# SINGLE-MATCH PARSER
# ---------------------------------------------------------------------------

def _parse_match(filepath: Path, alias: dict[str, str]) -> list[dict]:
    """
    Parse one Cricsheet JSON file into a list of delivery dicts.
    Returns an empty list if the file is malformed.
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [SKIP] {filepath.name}: {e}")
        return []

    info       = data.get("info", {})
    match_id   = filepath.stem
    dates      = info.get("dates", [])
    match_date = dates[0] if dates else ""
    season     = int(match_date[:4]) if match_date and match_date[:4].isdigit() else 0
    raw_venue  = info.get("venue", "")
    venue      = VENUE_FIX.get(raw_venue, raw_venue)
    teams      = info.get("teams", [])

    rows: list[dict] = []
    innings1_total = 0     # stored after innings 1 finishes to compute target

    for inning_idx, inning in enumerate(data.get("innings", [])):
        innings_num  = inning_idx + 1
        batting_team = inning.get("team", "")
        bowling_team = next((t for t in teams if t != batting_team), "")

        # Running totals reset for each innings
        score         = 0
        wickets       = 0
        legal_balls   = 0   # legal deliveries (wides excluded, no-balls included)

        for over_data in inning.get("overs", []):
            over     = over_data["over"]   # 0-indexed
            phase    = _phase(over)
            ball_in_over   = 0            # all deliveries in this over (1-indexed)
            legal_in_over  = 0            # legal deliveries in this over (for display)

            for delivery in over_data.get("deliveries", []):
                ball_in_over += 1

                batter       = _resolve(delivery.get("batter", ""),       alias)
                bowler       = _resolve(delivery.get("bowler", ""),       alias)
                non_striker  = _resolve(delivery.get("non_striker", ""),  alias)

                runs         = delivery.get("runs", {})
                runs_batter  = int(runs.get("batter",  0))
                runs_extras  = int(runs.get("extras",  0))
                runs_total   = int(runs.get("total",   0))

                extras       = delivery.get("extras", {})
                is_wide      = "wides"   in extras
                is_noball    = "noballs" in extras

                # Wide = not a legal delivery; no-ball IS (batter still faces it)
                if not is_wide:
                    legal_balls  += 1
                    legal_in_over += 1

                # Wickets
                wicket_list  = delivery.get("wickets", [])
                is_wicket    = len(wicket_list) > 0
                wicket_type  = wicket_list[0].get("kind", "") if wicket_list else ""

                # Update running totals
                score   += runs_total
                if is_wicket:
                    wickets += len(wicket_list)

                # Derived: balls remaining (legal balls basis)
                balls_remaining = max(0, 120 - legal_balls)
                overs_bowled    = legal_balls / 6
                crr             = round(score / overs_bowled, 2) if overs_bowled > 0 else 0.0

                # Innings-2 chase metrics
                if innings_num == 2:
                    target        = innings1_total + 1
                    required_runs = max(0, target - score)
                    rrr           = round(
                        required_runs / (balls_remaining / 6), 2
                    ) if balls_remaining > 0 else 99.9
                else:
                    target        = 0
                    required_runs = 0
                    rrr           = 0.0

                rows.append({
                    "match_id":        match_id,
                    "season":          season,
                    "date":            match_date,
                    "venue":           venue,
                    "batting_team":    batting_team,
                    "bowling_team":    bowling_team,
                    "innings":         innings_num,
                    "over":            over,
                    "ball":            ball_in_over,
                    "legal_ball":      legal_in_over,
                    "batter":          batter,
                    "bowler":          bowler,
                    "non_striker":     non_striker,
                    "runs_batter":     runs_batter,
                    "runs_extras":     runs_extras,
                    "runs_total":      runs_total,
                    "is_wide":         is_wide,
                    "is_noball":       is_noball,
                    "is_wicket":       is_wicket,
                    "wicket_type":     wicket_type,
                    "phase":           phase,
                    "innings_score":   score,
                    "innings_wickets": wickets,
                    "target":          target,
                    "required_runs":   required_runs,
                    "balls_remaining": balls_remaining,
                    "rrr":             rrr,
                    "crr":             crr,
                })

        # Save innings-1 final score so innings-2 has a target
        if innings_num == 1:
            innings1_total = score

    return rows


# ---------------------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------------------

def run(
    raw_dir:            Optional[Path] = None,
    output_path:        Optional[Path] = None,
    player_index_path:  Optional[Path] = None,
    verbose:            bool = True,
) -> pd.DataFrame:
    """
    Parse all Cricsheet JSON files and write ball_by_ball.parquet.

    Args:
        raw_dir:           Directory containing *.json match files.
                           Falls back to data/raw/ then ../psl_json/.
        output_path:       Destination parquet path. Default: data/processed/ball_by_ball.parquet
        player_index_path: Path to player_index.csv. Default: data/processed/ or ../
        verbose:           Print progress.

    Returns:
        The complete ball-by-ball DataFrame (also written to disk).
    """
    t0 = time.time()

    # ── Resolve raw data directory ──────────────────────────────────────────
    if raw_dir is None:
        if list(RAW_DIR.glob("*.json")):
            raw_dir = RAW_DIR
        elif RAW_FALLBACK.exists():
            raw_dir = RAW_FALLBACK
        else:
            raise FileNotFoundError(
                f"No JSON files found in {RAW_DIR} or {RAW_FALLBACK}. "
                "Place Cricsheet PSL JSON files in data/raw/ and re-run."
            )

    json_files = sorted(raw_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No *.json files found in {raw_dir}")

    if verbose:
        print(f"[parse_cricsheet] Found {len(json_files)} files in {raw_dir}")

    # ── Resolve player_index.csv ────────────────────────────────────────────
    if player_index_path is None:
        if PLAYER_INDEX_SPEC.exists():
            player_index_path = PLAYER_INDEX_SPEC
        elif PLAYER_INDEX_ACTUAL.exists():
            player_index_path = PLAYER_INDEX_ACTUAL
            # Copy to processed/ so subsequent pipeline steps find it there
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy(player_index_path, PLAYER_INDEX_SPEC)
            if verbose:
                print(f"[parse_cricsheet] Copied player_index.csv -> {PLAYER_INDEX_SPEC}")
        else:
            if verbose:
                print("[parse_cricsheet] WARNING: player_index.csv not found - no alias resolution")
            player_index_path = None

    alias = _load_alias_map(player_index_path) if player_index_path else {}
    if verbose and alias:
        print(f"[parse_cricsheet] Loaded {len(alias)} player name aliases")

    # ── Parse every match ───────────────────────────────────────────────────
    all_rows: list[dict] = []
    errors   = 0

    for i, fp in enumerate(json_files, 1):
        rows = _parse_match(fp, alias)
        if rows:
            all_rows.extend(rows)
        else:
            errors += 1
        if verbose and i % 50 == 0:
            print(f"  {i}/{len(json_files)} matches parsed...")

    if verbose:
        print(f"[parse_cricsheet] {len(json_files) - errors}/{len(json_files)} matches OK  "
              f"({errors} skipped)  ->  {len(all_rows):,} deliveries total")

    # ── Build DataFrame ─────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)

    # Enforce tidy dtypes
    df["season"]   = df["season"].astype("int16")
    df["over"]     = df["over"].astype("int8")
    df["innings"]  = df["innings"].astype("int8")
    df["is_wide"]  = df["is_wide"].astype(bool)
    df["is_noball"]= df["is_noball"].astype(bool)
    df["is_wicket"]= df["is_wicket"].astype(bool)
    df["phase"]    = df["phase"].astype("category")

    # Sort: season -> date → match → innings → over → ball
    df = df.sort_values(["season", "date", "match_id", "innings", "over", "ball"]) \
           .reset_index(drop=True)

    # ── Write parquet ────────────────────────────────────────────────────────
    if output_path is None:
        output_path = OUTPUT_FILE
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    elapsed = time.time() - t0
    if verbose:
        print(f"[parse_cricsheet] Written -> {output_path}  "
              f"({output_path.stat().st_size / 1024:.0f} KB)  "
              f"[{elapsed:.1f}s]")

    return df


# ---------------------------------------------------------------------------
# QUICK SANITY CHECK
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*55}")
    print(f"  ball_by_ball.parquet - summary")
    print(f"{'='*55}")
    print(f"  Rows          : {len(df):,}")
    print(f"  Columns       : {len(df.columns)}")
    print(f"  Seasons       : {sorted(df['season'].unique())}")
    print(f"  Matches       : {df['match_id'].nunique()}")
    print(f"  Venues        : {df['venue'].nunique()}")
    print(f"  Unique batters: {df['batter'].nunique()}")
    print(f"  Unique bowlers: {df['bowler'].nunique()}")
    print(f"  Wickets       : {df['is_wicket'].sum():,}")
    print(f"  Wides         : {df['is_wide'].sum():,}")
    print(f"  No-balls      : {df['is_noball'].sum():,}")
    print(f"\n  Phase breakdown:")
    for phase, cnt in df["phase"].value_counts().items():
        pct = cnt / len(df) * 100
        print(f"    {phase:<12} {cnt:>7,}  ({pct:.1f}%)")
    print(f"\n  Seasons × matches:")
    for season, grp in df.groupby("season"):
        matches = grp["match_id"].nunique()
        deliveries = len(grp)
        print(f"    {season}  {matches:>3} matches  {deliveries:>6,} deliveries")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = run(verbose=True)
    _print_summary(df)
