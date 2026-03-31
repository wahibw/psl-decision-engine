# pipeline/build_features.py
# Reads ball_by_ball.parquet and produces three derived feature tables:
#
#   data/processed/player_stats.parquet    -- per-player batting + bowling by phase/season
#   data/processed/venue_stats.csv         -- venue pitch profiles + historical averages
#   data/processed/matchup_matrix.parquet  -- batter vs bowler H2H (all PSL seasons)
#
# Run directly:  python pipeline/build_features.py
# Or import:     from pipeline.build_features import run

from __future__ import annotations

import csv
import time
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
PLAYER_INDEX  = PROCESSED_DIR / "player_index_2026_enriched.csv"

# psl_schedule.csv: spec location vs actual location
SCHEDULE_SPEC   = PROJ_ROOT / "data" / "psl_schedule.csv"
SCHEDULE_ACTUAL = PROJ_ROOT.parent / "psl_schedule.csv"

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _find_schedule() -> Optional[Path]:
    for p in (SCHEDULE_SPEC, SCHEDULE_ACTUAL):
        if p.exists():
            return p
    return None


def _load_bowling_styles(player_index_path: Path) -> dict[str, str]:
    """Returns {player_name: 'pace'|'spin'|'unknown'}."""
    styles: dict[str, str] = {}
    if not player_index_path.exists():
        return styles
    with open(player_index_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name  = row.get("player_name", "").strip()
            style = (row.get("bowling_style") or "").lower()
            role  = (row.get("primary_role")  or "").lower()
            if not name:
                continue
            if not style and any(w in role for w in ("batsman", "wk")):
                styles[name] = "non-bowler"
                continue
            if any(w in style for w in ("fast", "medium")):
                styles[name] = "pace"
            elif any(w in style for w in ("break", "orthodox", "spin", "leg")):
                styles[name] = "spin"
            else:
                styles[name] = "unknown"
    return styles


def _safe_divide(num: float, denom: float, default: float = 0.0) -> float:
    return round(num / denom, 2) if denom > 0 else default


# ---------------------------------------------------------------------------
# 1.  PLAYER STATS
# ---------------------------------------------------------------------------

def _build_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (player_name, season, phase).
    Contains both batting and bowling columns -- bowlers have NaN batting cols, etc.
    Adds a 'career' season (season=0) and 'overall' phase aggregation.
    """

    # ── BATTING ──────────────────────────────────────────────────────────────
    # Wides don't count as balls faced; batter column is reliable for all others
    bat_raw = df[~df["is_wide"]].copy()

    # Wickets attributable to the batter (exclude run outs of non-striker)
    bat_raw["batter_dismissed"] = bat_raw["is_wicket"] & (
        ~bat_raw["wicket_type"].isin(["run out", "obstructed the field", "retired hurt"])
    )

    def _bat_agg(grp: pd.DataFrame) -> pd.Series:
        innings   = grp["match_id"].nunique()
        balls     = len(grp)
        runs      = grp["runs_batter"].sum()
        dismissed = grp["batter_dismissed"].sum()
        fours     = (grp["runs_batter"] == 4).sum()
        sixes     = (grp["runs_batter"] == 6).sum()
        dots      = (grp["runs_batter"] == 0).sum()
        return pd.Series({
            "bat_innings":      innings,
            "bat_balls":        balls,
            "bat_runs":         runs,
            "bat_dismissals":   dismissed,
            "bat_avg":          _safe_divide(runs, dismissed, default=runs),
            "bat_sr":           _safe_divide(runs * 100, balls),
            "bat_4s":           int(fours),
            "bat_6s":           int(sixes),
            "bat_boundary_pct": _safe_divide((fours * 4 + sixes * 6) * 100, runs) if runs > 0 else 0.0,
            "bat_dot_pct":      _safe_divide(dots * 100, balls),
        })

    bat_by_phase  = bat_raw.groupby(["batter", "season", "phase"]).apply(_bat_agg).reset_index()
    bat_overall   = bat_raw.groupby(["batter", "season"]).apply(_bat_agg).reset_index()
    bat_overall["phase"] = "overall"
    bat_career    = bat_raw.groupby(["batter"]).apply(_bat_agg).reset_index()
    bat_career["season"] = 0
    bat_career["phase"]  = "overall"
    bat_career_phase = bat_raw.groupby(["batter", "phase"]).apply(_bat_agg).reset_index()
    bat_career_phase["season"] = 0

    bat_all = pd.concat([bat_by_phase, bat_overall, bat_career, bat_career_phase], ignore_index=True)
    bat_all = bat_all.rename(columns={"batter": "player_name"})

    # ── CHASE / SET BATTING CONTEXT ──────────────────────────────────────────
    # bat_avg_chase / bat_sr_chase  — innings == 2 (chasing a target)
    # bat_avg_set   / bat_sr_set    — innings == 1 (setting a target)
    # innings_context_split         — chase_avg minus set_avg (positive = better chaser)
    #
    # Populated only on phase == "overall" rows (both per-season and career season=0).
    # All other phase rows (powerplay, middle, death) carry NaN for these columns.

    def _context_agg(grp: pd.DataFrame) -> pd.Series:
        runs      = int(grp["runs_batter"].sum())
        dismissed = int(grp["batter_dismissed"].sum())
        balls     = len(grp)
        innings_n = int(grp["match_id"].nunique())
        return pd.Series({
            "ctx_runs":      runs,
            "ctx_dismissed": dismissed,
            "ctx_balls":     balls,
            "ctx_innings":   innings_n,
        })

    def _build_context(bat_sub: pd.DataFrame, suffix: str, career: bool) -> pd.DataFrame:
        """
        Returns (player_name, season, bat_avg_{suffix}, bat_sr_{suffix}, bat_innings_{suffix}).
        career=True  → group by batter only, season = 0
        career=False → group by (batter, season)
        """
        if career:
            agg = bat_sub.groupby("batter").apply(_context_agg).reset_index()
            agg["season"] = 0
        else:
            agg = bat_sub.groupby(["batter", "season"]).apply(_context_agg).reset_index()

        agg = agg.rename(columns={"batter": "player_name"})
        agg[f"bat_avg_{suffix}"] = agg.apply(
            lambda r: _safe_divide(r["ctx_runs"], r["ctx_dismissed"],
                                   default=float(r["ctx_runs"])),
            axis=1,
        )
        agg[f"bat_sr_{suffix}"]      = agg.apply(
            lambda r: _safe_divide(r["ctx_runs"] * 100, r["ctx_balls"]), axis=1,
        )
        agg[f"bat_innings_{suffix}"] = agg["ctx_innings"]
        return agg[["player_name", "season",
                    f"bat_avg_{suffix}", f"bat_sr_{suffix}", f"bat_innings_{suffix}"]]

    bat_set_season   = _build_context(bat_raw[bat_raw["innings"] == 1], "set",   career=False)
    bat_set_career   = _build_context(bat_raw[bat_raw["innings"] == 1], "set",   career=True)
    bat_chase_season = _build_context(bat_raw[bat_raw["innings"] == 2], "chase", career=False)
    bat_chase_career = _build_context(bat_raw[bat_raw["innings"] == 2], "chase", career=True)

    set_all   = pd.concat([bat_set_season,   bat_set_career],   ignore_index=True)
    chase_all = pd.concat([bat_chase_season, bat_chase_career], ignore_index=True)

    context_df = set_all.merge(chase_all, on=["player_name", "season"], how="outer")
    context_df["innings_context_split"] = (
        context_df["bat_avg_chase"].fillna(0) - context_df["bat_avg_set"].fillna(0)
    )

    _CTX_COLS = [
        "bat_avg_chase", "bat_sr_chase", "bat_innings_chase",
        "bat_avg_set",   "bat_sr_set",   "bat_innings_set",
        "innings_context_split",
    ]

    bat_all = bat_all.merge(context_df[["player_name", "season"] + _CTX_COLS],
                            on=["player_name", "season"], how="left")
    # Only 'overall' phase rows carry context values; phase-specific rows get NaN
    non_overall = bat_all["phase"] != "overall"
    bat_all.loc[non_overall, _CTX_COLS] = float("nan")

    # ── BOWLING ──────────────────────────────────────────────────────────────
    # All deliveries including wides/no-balls (they count against bowler)
    # But legal_ball is used for economy (wides don't count as legal)
    bowl_raw = df.copy()

    # Wickets credited to bowler: exclude run outs
    bowl_raw["bowl_wicket"] = bowl_raw["is_wicket"] & (
        ~bowl_raw["wicket_type"].isin(["run out", "obstructed the field", "retired hurt"])
    )
    bowl_raw["is_boundary"] = bowl_raw["runs_batter"].isin([4, 6])

    def _bowl_agg(grp: pd.DataFrame) -> pd.Series:
        legal_balls = (~grp["is_wide"]).sum()   # wides don't count as legal
        total_runs  = grp["runs_total"].sum()
        wickets     = grp["bowl_wicket"].sum()
        dots        = ((~grp["is_wide"]) & (grp["runs_total"] == 0)).sum()
        boundaries  = grp["is_boundary"].sum()
        overs       = round(legal_balls / 6, 1)
        return pd.Series({
            "bowl_balls":        legal_balls,
            "bowl_overs":        overs,
            "bowl_runs":         int(total_runs),
            "bowl_wickets":      int(wickets),
            "bowl_economy":      _safe_divide(total_runs * 6, legal_balls),
            "bowl_sr":           _safe_divide(legal_balls, wickets, default=99.9),
            "bowl_avg":          _safe_divide(total_runs, wickets, default=total_runs),
            "bowl_dot_pct":      _safe_divide(dots * 100, legal_balls),
            "bowl_boundary_pct": _safe_divide(boundaries * 100, legal_balls),
        })

    bowl_by_phase  = bowl_raw.groupby(["bowler", "season", "phase"]).apply(_bowl_agg).reset_index()
    bowl_overall   = bowl_raw.groupby(["bowler", "season"]).apply(_bowl_agg).reset_index()
    bowl_overall["phase"] = "overall"
    bowl_career    = bowl_raw.groupby(["bowler"]).apply(_bowl_agg).reset_index()
    bowl_career["season"] = 0
    bowl_career["phase"]  = "overall"
    bowl_career_phase = bowl_raw.groupby(["bowler", "phase"]).apply(_bowl_agg).reset_index()
    bowl_career_phase["season"] = 0

    bowl_all = pd.concat([bowl_by_phase, bowl_overall, bowl_career, bowl_career_phase], ignore_index=True)
    bowl_all = bowl_all.rename(columns={"bowler": "player_name"})

    # ── MERGE batting + bowling on player/season/phase ────────────────────
    merged = pd.merge(
        bat_all, bowl_all,
        on=["player_name", "season", "phase"],
        how="outer",
    )
    merged["season"] = merged["season"].astype(int)

    return merged.sort_values(["player_name", "season", "phase"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2.  VENUE STATS
# ---------------------------------------------------------------------------

def _build_venue_stats(
    df: pd.DataFrame,
    schedule_path: Optional[Path],
    bowling_styles: dict[str, str],
) -> pd.DataFrame:
    """
    One row per venue with batting and bowling aggregates.
    Joins psl_schedule.csv to compute chase/defend win percentages.
    """
    rows = []

    # Load schedule for match outcomes
    outcomes: dict[str, dict] = {}   # match_id -> {winner, team1, team2}
    if schedule_path:
        with open(schedule_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                # Infer match_id from date + team combo — use match_number+date as key
                # We'll join on (date, team1, team2) since match_id in schedule isn't the same format
                key = (row["match_date"], row["team1"], row["team2"])
                outcomes[key] = row

    # Build a match-level lookup: match_id -> winner, teams
    # ball_by_ball has match_id (= JSON filename stem) + date + batting_team
    match_meta = (
        df.groupby("match_id")
        .agg(
            date=("date", "first"),
            team1=("batting_team", lambda x: x[df.loc[x.index, "innings"] == 1].iloc[0]
                                  if (df.loc[x.index, "innings"] == 1).any() else x.iloc[0]),
            team2=("batting_team", lambda x: x[df.loc[x.index, "innings"] == 2].iloc[0]
                                  if (df.loc[x.index, "innings"] == 2).any() else ""),
        )
        .reset_index()
    )

    # Derive chase/defend from ball_by_ball: innings-2 last delivery
    inn2_last = (
        df[df["innings"] == 2]
        .sort_values(["match_id", "innings", "over", "ball"])
        .groupby("match_id")
        .last()[["innings_score", "target", "innings_wickets", "season"]]
        .reset_index()
    )
    inn2_last["chase_won"] = inn2_last["innings_score"] >= inn2_last["target"]

    match_outcome = dict(zip(inn2_last["match_id"], inn2_last["chase_won"]))
    # season lookup used for recency-weighted chase_win_pct below
    match_season  = dict(zip(inn2_last["match_id"], inn2_last["season"]))

    for venue, vdf in df.groupby("venue"):
        match_ids = vdf["match_id"].unique()
        n_matches = len(match_ids)

        # First innings scores at this venue
        inn1 = vdf[vdf["innings"] == 1]
        inn2 = vdf[vdf["innings"] == 2]

        inn1_scores = inn1.groupby("match_id")["innings_score"].max()
        inn2_scores = inn2.groupby("match_id")["innings_score"].max()

        # Phase scores: powerplay and death (first innings only for comparison)
        pp_scores  = inn1[inn1["phase"] == "powerplay"].groupby("match_id")["innings_score"].max()
        dth_scores = inn1[inn1["phase"] == "death"].groupby("match_id")["innings_score"].max()
        dth_start  = inn1[inn1["phase"] == "death"].groupby("match_id")["innings_score"].min()
        death_runs_added = (dth_scores - dth_start).clip(lower=0)

        # Wickets per innings
        wickets_per_match = vdf.groupby(["match_id", "innings"])["is_wicket"].sum()
        avg_wickets = wickets_per_match.mean()

        # Chase/defend — recency-weighted to avoid bubble-season distortion.
        # Last 3 seasons are counted twice; older seasons counted once.
        # Falls back to flat all-time average when recent sample < 10 matches.
        venue_match_ids = set(match_ids)
        all_seasons = sorted({match_season[k] for k in venue_match_ids if k in match_season}, reverse=True)
        recent_cutoff = all_seasons[2] if len(all_seasons) >= 3 else (all_seasons[0] if all_seasons else 0)

        weighted_wins = 0.0
        weighted_total = 0.0
        recent_total = 0
        for mid in venue_match_ids:
            if mid not in match_outcome:
                continue
            won    = match_outcome[mid]
            season = match_season.get(mid, 0)
            weight = 2.0 if season >= recent_cutoff else 1.0
            weighted_wins  += won * weight
            weighted_total += weight
            if season >= recent_cutoff:
                recent_total += 1

        if recent_total >= 10:
            # Enough recent matches — use recency-weighted figure
            chase_win_pct = _safe_divide(weighted_wins * 100, weighted_total)
        else:
            # Sparse recent data — fall back to flat all-time average
            chase_results = [v for k, v in match_outcome.items() if k in venue_match_ids]
            chase_win_pct = _safe_divide(sum(chase_results) * 100, len(chase_results))

        defend_win_pct = 100.0 - chase_win_pct if weighted_total > 0 else 0.0

        # Economy by bowling type
        venue_bowlers = vdf[~vdf["is_wide"]].copy()
        venue_bowlers["bowl_style"] = venue_bowlers["bowler"].map(
            lambda b: bowling_styles.get(b, "unknown")
        )
        def _economy(sub: pd.DataFrame) -> float:
            legal = len(sub)
            runs  = sub["runs_total"].sum()
            return _safe_divide(runs * 6, legal)

        pace_eco = _economy(venue_bowlers[venue_bowlers["bowl_style"] == "pace"])
        spin_eco = _economy(venue_bowlers[venue_bowlers["bowl_style"] == "spin"])

        # Overall run rate
        legal_all = (~vdf["is_wide"]).sum()
        total_runs = vdf["runs_total"].sum()

        rows.append({
            "venue":                  venue,
            "matches":                n_matches,
            "avg_first_score":        round(inn1_scores.mean(), 1) if len(inn1_scores) else 0,
            "avg_second_score":       round(inn2_scores.mean(), 1) if len(inn2_scores) else 0,
            "avg_pp_score":           round(pp_scores.mean(), 1)   if len(pp_scores)  else 0,
            "avg_death_runs_added":   round(death_runs_added.mean(), 1) if len(death_runs_added) else 0,
            "chase_win_pct":          chase_win_pct,
            "defend_win_pct":         defend_win_pct,
            "avg_wickets_per_innings":round(avg_wickets, 1),
            "avg_runs_per_over":      _safe_divide(total_runs * 6, legal_all),
            "pace_economy":           pace_eco,
            "spin_economy":           spin_eco,
        })

    return pd.DataFrame(rows).sort_values("matches", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3.  MATCHUP MATRIX
# ---------------------------------------------------------------------------

def _build_matchup_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (batter, bowler) pair with full H2H stats.
    Only legal deliveries (non-wide) included in facing stats.
    """
    h2h = df[~df["is_wide"]].copy()

    h2h["bowl_wicket"] = h2h["is_wicket"] & (
        ~h2h["wicket_type"].isin(["run out", "obstructed the field", "retired hurt"])
    )
    h2h["is_boundary"] = h2h["runs_batter"].isin([4, 6])
    h2h["is_dot"]      = h2h["runs_batter"] == 0

    agg = h2h.groupby(["batter", "bowler"]).agg(
        balls         =("runs_batter",  "count"),
        runs          =("runs_batter",  "sum"),
        dismissals    =("bowl_wicket",  "sum"),
        fours         =("runs_batter",  lambda x: (x == 4).sum()),
        sixes         =("runs_batter",  lambda x: (x == 6).sum()),
        dots          =("is_dot",       "sum"),
        boundaries    =("is_boundary",  "sum"),
        seasons_active=("season",       "nunique"),
        matches_faced =("match_id",     "nunique"),
    ).reset_index()

    agg["sr"]             = (agg["runs"] * 100 / agg["balls"]).round(1)
    agg["dismissal_pct"]  = (agg["dismissals"] * 100 / agg["balls"]).round(2)
    agg["dot_pct"]        = (agg["dots"] * 100 / agg["balls"]).round(1)
    agg["boundary_pct"]   = (agg["boundaries"] * 100 / agg["balls"]).round(1)
    # Normalize both metrics to comparable scale.
    # dismissal_pct/100 = fraction of balls ending in dismissal (0.0 to ~0.3)
    # sr/150 = batter SR relative to T20 baseline of 150 (0.5=SR75, 1.0=SR150, 1.3=SR200)
    # Positive value = bowler dominant. Negative = batter dominant.
    agg["bowler_adv"] = (agg["dismissal_pct"] / 100) - (agg["sr"] / 150)

    return agg.sort_values(["batter", "bowler"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------------------

def run(
    bbb_path:     Optional[Path] = None,
    output_dir:   Optional[Path] = None,
    verbose:      bool = True,
) -> dict[str, pd.DataFrame | Path]:
    """
    Build all three feature tables from ball_by_ball.parquet.

    Returns:
        dict with keys 'player_stats', 'venue_stats', 'matchup_matrix'
        and 'player_stats_path', 'venue_stats_path', 'matchup_matrix_path'.
    """
    t0 = time.time()

    bbb_path   = Path(bbb_path or BBB_FILE)
    output_dir = Path(output_dir or PROCESSED_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not bbb_path.exists():
        raise FileNotFoundError(
            f"ball_by_ball.parquet not found at {bbb_path}. "
            "Run pipeline/parse_cricsheet.py first."
        )

    if verbose:
        print(f"[build_features] Loading {bbb_path.name}...")

    df = pd.read_parquet(bbb_path)
    if verbose:
        print(f"[build_features] {len(df):,} rows, {df['match_id'].nunique()} matches")

    # Load supporting tables
    bowling_styles = _load_bowling_styles(PLAYER_INDEX)
    if verbose:
        print(f"[build_features] Bowling styles loaded: {len(bowling_styles)} players")

    schedule_path = _find_schedule()
    if verbose:
        if schedule_path:
            print(f"[build_features] Schedule found: {schedule_path.name}")
        else:
            print("[build_features] WARNING: psl_schedule.csv not found - chase/defend stats will be incomplete")

    # ── 1. Player stats ──────────────────────────────────────────────────────
    if verbose:
        print("[build_features] Building player_stats...")
    player_stats = _build_player_stats(df)
    ps_path = output_dir / "player_stats.parquet"
    player_stats.to_parquet(ps_path, index=False)
    if verbose:
        print(f"  player_stats: {len(player_stats):,} rows -> {ps_path.name}")
        _print_chase_set_sanity(player_stats)

    # ── 2. Venue stats ───────────────────────────────────────────────────────
    if verbose:
        print("[build_features] Building venue_stats...")
    venue_stats = _build_venue_stats(df, schedule_path, bowling_styles)
    vs_path = output_dir / "venue_stats.csv"
    venue_stats.to_csv(vs_path, index=False)
    if verbose:
        print(f"  venue_stats: {len(venue_stats)} venues -> {vs_path.name}")

    # ── 3. Matchup matrix ────────────────────────────────────────────────────
    if verbose:
        print("[build_features] Building matchup_matrix...")
    matchup = _build_matchup_matrix(df)
    mm_path = output_dir / "matchup_matrix.parquet"
    matchup.to_parquet(mm_path, index=False)
    if verbose:
        print(f"  matchup_matrix: {len(matchup):,} pairs -> {mm_path.name}")

    elapsed = time.time() - t0
    if verbose:
        print(f"[build_features] Done in {elapsed:.1f}s")

    return {
        "player_stats":         player_stats,
        "venue_stats":          venue_stats,
        "matchup_matrix":       matchup,
        "player_stats_path":    ps_path,
        "venue_stats_path":     vs_path,
        "matchup_matrix_path":  mm_path,
    }


# ---------------------------------------------------------------------------
# CHASE / SET SANITY CHECK
# ---------------------------------------------------------------------------

def _print_chase_set_sanity(player_stats: pd.DataFrame) -> None:
    """
    Print top 5 PSL chasers and top 5 setters by innings_context_split.
    Requires >=5 innings in each context to appear in the lists.
    """
    career = player_stats[
        (player_stats["season"] == 0) &
        (player_stats["phase"]  == "overall") &
        player_stats["innings_context_split"].notna()
    ].copy()

    # Minimum 5 innings in each context for a reliable split
    MIN_INN = 5
    qualified = career[
        (career["bat_innings_chase"].fillna(0) >= MIN_INN) &
        (career["bat_innings_set"].fillna(0)   >= MIN_INN)
    ].copy()

    if qualified.empty:
        print("  [chase/set] No players with >=5 innings in both contexts yet.")
        return

    cols = ["player_name", "bat_avg_chase", "bat_avg_set",
            "bat_sr_chase", "bat_sr_set", "innings_context_split",
            "bat_innings_chase", "bat_innings_set"]

    top_chasers = qualified.nlargest(5, "innings_context_split")[cols]
    top_setters = qualified.nsmallest(5, "innings_context_split")[cols]

    print(f"\n  Chase vs Set split (career, >=5 innings each context)")
    print(f"  {'Player':<25} {'Avg Chase':>9} {'Avg Set':>8} {'SR Chase':>9} {'SR Set':>8} {'Split':>7}")
    print(f"  {'-'*70}")
    print("  --- Top 5 Chasers (positive split = better in chases) ---")
    for _, r in top_chasers.iterrows():
        print(f"  {r['player_name']:<25} "
              f"{r['bat_avg_chase']:>9.1f} {r['bat_avg_set']:>8.1f} "
              f"{r['bat_sr_chase']:>9.1f} {r['bat_sr_set']:>8.1f} "
              f"{r['innings_context_split']:>+7.1f}")
    print("  --- Top 5 Setters (negative split = better in set innings) ---")
    for _, r in top_setters.iterrows():
        print(f"  {r['player_name']:<25} "
              f"{r['bat_avg_chase']:>9.1f} {r['bat_avg_set']:>8.1f} "
              f"{r['bat_sr_chase']:>9.1f} {r['bat_sr_set']:>8.1f} "
              f"{r['innings_context_split']:>+7.1f}")


# ---------------------------------------------------------------------------
# ENTRY POINT + SUMMARY
# ---------------------------------------------------------------------------

def _print_summary(results: dict) -> None:
    ps = results["player_stats"]
    vs = results["venue_stats"]
    mm = results["matchup_matrix"]

    print(f"\n{'='*55}")
    print("  build_features -- summary")
    print(f"{'='*55}")

    print(f"\n  player_stats ({len(ps):,} rows)")
    print(f"    Unique players : {ps['player_name'].nunique()}")
    print(f"    Seasons covered: {sorted(ps['season'].unique())}")
    print(f"    Phases         : {sorted(ps['phase'].unique())}")
    _print_chase_set_sanity(ps)

    print(f"\n  venue_stats ({len(vs)} venues)")
    for _, row in vs.iterrows():
        print(f"    {row['venue'][:40]:<40}  "
              f"avg1st={row['avg_first_score']:>5.1f}  "
              f"chase%={row['chase_win_pct']:>5.1f}  "
              f"pace_eco={row['pace_economy']:>4.2f}  "
              f"spin_eco={row['spin_economy']:>4.2f}")

    print(f"\n  matchup_matrix ({len(mm):,} pairs)")
    qual = mm[mm["balls"] >= 8]
    print(f"    Pairs with >=8 balls : {len(qual):,}")
    print(f"    Pairs with >=20 balls: {len(mm[mm['balls'] >= 20]):,}")
    top = qual.nlargest(5, "dismissal_pct")[["batter","bowler","balls","runs","dismissals","sr","dismissal_pct"]]
    print(f"\n    Top 5 bowler-dominant matchups (>=8 balls):")
    for _, r in top.iterrows():
        print(f"      {r['bowler']:<25} vs {r['batter']:<25} "
              f"{r['balls']}b {r['runs']}r {r['dismissals']}w  "
              f"SR={r['sr']} dis%={r['dismissal_pct']}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    results = run(verbose=True)
    _print_summary(results)
