# pipeline/build_recent_form.py
# Recent form feature builder: reads ball_by_ball.parquet, computes per-player
# rolling form metrics across their last 10 T20 matches, and writes
# data/processed/recent_form.parquet.
#
# Output schema  (one row per player×venue, plus one row with venue="" for overall):
#   player_name, reference_date, last_10_matches,
#   bat_runs_total, bat_innings, bat_avg, bat_sr, bat_boundary_pct, bat_dot_pct,
#   bat_form_score (0-100), bat_trend,
#   bowl_overs, bowl_wickets, bowl_economy, bowl_dot_pct, bowl_avg, bowl_sr,
#   bowl_form_score (0-100), bowl_trend,
#   venue, venue_bat_avg, venue_bat_sr, venue_bowl_economy, venue_bowl_wickets,
#   venue_matches, venue_form_score,
#   dew_condition_bat_sr, dew_condition_bowl_eco,
#   high_humidity_bowl_eco, night_match_bat_sr
#
# Usage:
#   python pipeline/build_recent_form.py
#   python pipeline/build_recent_form.py --quiet

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJ_ROOT   = Path(__file__).resolve().parent.parent
BBB_PATH    = PROJ_ROOT / "data" / "processed" / "ball_by_ball.parquet"
OUTPUT_PATH = PROJ_ROOT / "data" / "processed" / "recent_form.parquet"

MIN_MATCHES       = 5   # min matches for a valid form score (else defaults to 50)
MIN_VENUE_MATCHES = 3   # min venue-specific matches for venue_form_score

# PSL venues with high dew probability (northern subcontinent, night matches Feb-Mar)
DEW_VENUES = {
    "Gaddafi Stadium, Lahore",
    "National Stadium, Karachi",
    "Pindi Cricket Stadium, Rawalpindi",
    "Multan Cricket Stadium",
}

# Coastal / high-humidity venues
HUMID_VENUES = {
    "National Stadium, Karachi",
}


# ---------------------------------------------------------------------------
# FORM SCORE FORMULAS
# ---------------------------------------------------------------------------

def _bat_form_score(avg: float, sr: float, boundary_pct: float, innings: int) -> float:
    """
    Batting form score 0-100.
    Full marks: avg=40, sr=170, boundary_pct=45%.
    Neutral (score≈50): avg=25, sr=135, boundary_pct=30%.
    Returns 50 when insufficient data (innings < MIN_MATCHES).
    """
    if innings < MIN_MATCHES:
        return 50.0
    avg_c      = min(1.0, avg / 40.0)            * 40.0
    sr_c       = min(1.0, sr  / 170.0)           * 40.0
    boundary_c = min(1.0, boundary_pct / 45.0)   * 20.0
    return round(avg_c + sr_c + boundary_c, 2)


def _bowl_form_score(eco: float, overs: float, wickets: int, dot_pct: float) -> float:
    """
    Bowling form score 0-100.
    Full marks: eco=6.0, wpo=0.5 per over, dot_pct=50%.
    Neutral (score≈50): eco=9.0, wpo=0.25, dot_pct=35%.
    Returns 50 when insufficient data (overs < MIN_MATCHES).
    """
    if overs < MIN_MATCHES:
        return 50.0
    eco_c  = max(0.0, min(1.0, (12.0 - eco) / 6.0)) * 40.0    # 6→40pts, 12→0pts
    wpo    = wickets / max(0.1, overs)
    wpo_c  = min(1.0, wpo / 0.5)                    * 35.0    # 0.5/over → 35pts
    dot_c  = min(1.0, dot_pct / 50.0)               * 25.0    # 50% → 25pts
    return round(eco_c + wpo_c + dot_c, 2)


def _venue_form_score(
    v_bat_avg: float, v_bat_sr: float,
    v_bowl_eco: float,
    v_matches: int,
    has_bat: bool, has_bowl: bool,
) -> float:
    """Venue-specific form score 0-100. Returns 50 if insufficient data."""
    if v_matches < MIN_VENUE_MATCHES:
        return 50.0
    bat_s  = (min(1.0, v_bat_avg / 40.0) * 40.0 + min(1.0, v_bat_sr / 170.0) * 60.0) if has_bat else 50.0
    bowl_s = max(0.0, min(1.0, (12.0 - v_bowl_eco) / 6.0)) * 100.0 if has_bowl else 50.0

    if has_bat and has_bowl:
        return round((bat_s + bowl_s) / 2.0, 2)
    if has_bat:
        return round(bat_s, 2)
    return round(bowl_s, 2)


# ---------------------------------------------------------------------------
# TREND
# ---------------------------------------------------------------------------

def _bat_trend(sorted_match_ids: list, bat_df: pd.DataFrame) -> str:
    """Compare SR of the older half vs newer half of the match window."""
    if len(sorted_match_ids) < 6:
        return "stable"
    mid = len(sorted_match_ids) // 2
    old_ids    = sorted_match_ids[:mid]
    recent_ids = sorted_match_ids[mid:]

    def _sr(ids: list) -> float:
        sub   = bat_df[bat_df["match_id"].isin(ids)]
        runs  = sub["runs_batter"].sum()
        balls = len(sub[sub["is_wide"] == 0]) if "is_wide" in sub.columns else len(sub)
        return (runs / max(1, balls)) * 100.0

    old_sr    = _sr(old_ids)
    recent_sr = _sr(recent_ids)
    if recent_sr > old_sr * 1.10:
        return "rising"
    if recent_sr < old_sr * 0.90:
        return "declining"
    return "stable"


def _bowl_trend(sorted_match_ids: list, bowl_df: pd.DataFrame) -> str:
    """Compare economy of the older half vs newer half (lower eco = improving = 'rising')."""
    if len(sorted_match_ids) < 6:
        return "stable"
    mid = len(sorted_match_ids) // 2
    old_ids    = sorted_match_ids[:mid]
    recent_ids = sorted_match_ids[mid:]

    def _eco(ids: list) -> float:
        sub   = bowl_df[bowl_df["match_id"].isin(ids)]
        if sub.empty:
            return 8.0
        runs  = sub["runs_total"].sum()
        legal = len(sub[(sub["is_wide"] == 0) & (sub["is_noball"] == 0)]) if \
                ("is_wide" in sub.columns and "is_noball" in sub.columns) else len(sub)
        overs = legal / 6.0
        return runs / max(0.1, overs)

    old_eco    = _eco(old_ids)
    recent_eco = _eco(recent_ids)
    if recent_eco < old_eco * 0.90:
        return "rising"     # improving — lower economy
    if recent_eco > old_eco * 1.10:
        return "declining"  # worsening — higher economy
    return "stable"


# ---------------------------------------------------------------------------
# STAT AGGREGATORS
# ---------------------------------------------------------------------------

def _bat_stats(df: pd.DataFrame) -> dict:
    """Batting stats from a filtered ball_by_ball subset (batter rows only)."""
    if df.empty:
        return {}
    is_wide_col = df["is_wide"] if "is_wide" in df.columns else pd.Series(0, index=df.index)
    legal_mask  = is_wide_col == 0
    legal_df    = df[legal_mask]
    balls_faced = int(legal_mask.sum())
    if balls_faced == 0:
        return {}

    runs        = int(df["runs_batter"].sum())
    innings     = df.groupby(["match_id", "innings"]).ngroups
    dismissals  = int(df["is_wicket"].sum())
    boundaries  = int((df["runs_batter"] >= 4).sum())
    dots        = int((legal_df["runs_batter"] == 0).sum())

    sr           = (runs / balls_faced) * 100.0
    avg          = runs / dismissals if dismissals > 0 else float(runs)  # not out → total runs
    boundary_pct = (boundaries / balls_faced) * 100.0
    dot_pct      = (dots / balls_faced) * 100.0

    return {
        "bat_runs_total":   runs,
        "bat_innings":      innings,
        "bat_avg":          round(avg, 2),
        "bat_sr":           round(sr, 2),
        "bat_boundary_pct": round(boundary_pct, 2),
        "bat_dot_pct":      round(dot_pct, 2),
    }


def _bowl_stats(df: pd.DataFrame) -> dict:
    """Bowling stats from a filtered ball_by_ball subset (bowler rows only)."""
    if df.empty:
        return {}
    is_wide_col   = df["is_wide"]   if "is_wide"   in df.columns else pd.Series(0, index=df.index)
    is_noball_col = df["is_noball"] if "is_noball" in df.columns else pd.Series(0, index=df.index)
    legal_mask    = (is_wide_col == 0) & (is_noball_col == 0)
    legal_balls   = int(legal_mask.sum())
    if legal_balls == 0:
        return {}

    runs_conceded = int(df["runs_total"].sum())
    wickets       = int(df["is_wicket"].sum())
    overs         = legal_balls / 6.0
    economy       = runs_conceded / overs
    legal_df      = df[legal_mask]
    dots          = int((legal_df["runs_total"] == 0).sum())
    dot_pct       = (dots / legal_balls) * 100.0
    bowl_avg      = rounds_safe(runs_conceded / wickets) if wickets > 0 else 999.0
    bowl_sr_val   = rounds_safe(legal_balls / wickets)   if wickets > 0 else 999.0

    return {
        "bowl_overs":   round(overs, 2),
        "bowl_wickets": wickets,
        "bowl_economy": round(economy, 2),
        "bowl_dot_pct": round(dot_pct, 2),
        "bowl_avg":     round(bowl_avg, 2),
        "bowl_sr":      round(bowl_sr_val, 2),
    }


def rounds_safe(v: float) -> float:
    return round(v, 2)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run(verbose: bool = True) -> None:
    """Build recent_form.parquet from ball_by_ball.parquet."""
    if verbose:
        print("  build_recent_form: loading ball_by_ball.parquet ...")

    if not BBB_PATH.exists():
        raise FileNotFoundError(f"ball_by_ball.parquet not found: {BBB_PATH}")

    bbb = pd.read_parquet(BBB_PATH)

    # Ensure date is datetime
    if bbb["date"].dtype == object:
        bbb["date"] = pd.to_datetime(bbb["date"], errors="coerce")

    # Sort chronologically (oldest first) for trend splitting
    bbb = bbb.sort_values(
        ["date", "match_id", "innings", "over", "ball"]
    ).reset_index(drop=True)

    # match_id → earliest date mapping for sorting
    match_dates = bbb.groupby("match_id")["date"].first()

    ref_date = str(date.today())

    all_batters = set(bbb["batter"].dropna().unique())
    all_bowlers = set(bbb["bowler"].dropna().unique())
    all_players = all_batters | all_bowlers

    if verbose:
        print(f"  Players found: {len(all_players)}")

    rows: list[dict] = []

    for player in sorted(all_players):
        bat_mask    = bbb["batter"] == player
        bowl_mask   = bbb["bowler"] == player
        appear_mask = bat_mask | bowl_mask

        match_ids = bbb.loc[appear_mask, "match_id"].unique()
        if len(match_ids) == 0:
            continue

        # Sort chronologically; take the 10 most recent
        sorted_ids = sorted(match_ids, key=lambda m: match_dates.get(m, pd.Timestamp.min))
        last10     = sorted_ids[-10:]
        n_matches  = len(last10)

        # ── Overall form (last 10 matches, all venues) ────────────────────
        bat_df_all  = bbb[bbb["match_id"].isin(last10) & bat_mask]
        bowl_df_all = bbb[bbb["match_id"].isin(last10) & bowl_mask]

        b_stats = _bat_stats(bat_df_all)
        w_stats = _bowl_stats(bowl_df_all)

        bat_innings = b_stats.get("bat_innings", 0)
        bat_avg_val = b_stats.get("bat_avg", 0.0)
        bat_sr_val  = b_stats.get("bat_sr", 0.0)
        bat_bpct    = b_stats.get("bat_boundary_pct", 0.0)
        bat_dpct    = b_stats.get("bat_dot_pct", 0.0)

        bowl_overs_val = w_stats.get("bowl_overs", 0.0)
        bowl_eco_val   = w_stats.get("bowl_economy", 0.0)
        bowl_wkts_val  = w_stats.get("bowl_wickets", 0)
        bowl_dpct_val  = w_stats.get("bowl_dot_pct", 0.0)

        bat_form  = _bat_form_score(bat_avg_val, bat_sr_val, bat_bpct, bat_innings)
        bowl_form = _bowl_form_score(bowl_eco_val, bowl_overs_val, bowl_wkts_val, bowl_dpct_val)
        bat_t     = _bat_trend(last10, bat_df_all)
        bowl_t    = _bowl_trend(last10, bowl_df_all)

        # ── Weather-condition splits ──────────────────────────────────────
        def _match_ids_at(venues: set) -> list:
            return list(bbb.loc[
                bbb["match_id"].isin(last10) & bbb["venue"].isin(venues), "match_id"
            ].unique())

        dew_ids   = _match_ids_at(DEW_VENUES)
        humid_ids = _match_ids_at(HUMID_VENUES)

        dew_bat  = _bat_stats( bbb[bbb["match_id"].isin(dew_ids)   & bat_mask])
        dew_bowl = _bowl_stats(bbb[bbb["match_id"].isin(dew_ids)   & bowl_mask])
        hmd_bowl = _bowl_stats(bbb[bbb["match_id"].isin(humid_ids) & bowl_mask])

        dew_bat_sr        = dew_bat.get( "bat_sr",        bat_sr_val)
        dew_bowl_eco      = dew_bowl.get("bowl_economy",  bowl_eco_val)
        humid_bowl_eco    = hmd_bowl.get("bowl_economy",  bowl_eco_val)
        night_bat_sr      = bat_sr_val   # all PSL matches are night matches

        # ── Overall row (venue = "") ──────────────────────────────────────
        overall_row = {
            "player_name":            player,
            "reference_date":         ref_date,
            "last_10_matches":        n_matches,
            "venue":                  "",
            # Batting
            "bat_runs_total":         b_stats.get("bat_runs_total", 0),
            "bat_innings":            bat_innings,
            "bat_avg":                bat_avg_val,
            "bat_sr":                 bat_sr_val,
            "bat_boundary_pct":       bat_bpct,
            "bat_dot_pct":            bat_dpct,
            "bat_form_score":         bat_form,
            "bat_trend":              bat_t,
            # Bowling
            "bowl_overs":             bowl_overs_val,
            "bowl_wickets":           bowl_wkts_val,
            "bowl_economy":           bowl_eco_val,
            "bowl_dot_pct":           bowl_dpct_val,
            "bowl_avg":               w_stats.get("bowl_avg", 999.0),
            "bowl_sr":                w_stats.get("bowl_sr",  999.0),
            "bowl_form_score":        bowl_form,
            "bowl_trend":             bowl_t,
            # Venue-specific (empty for overall row)
            "venue_bat_avg":          0.0,
            "venue_bat_sr":           0.0,
            "venue_bowl_economy":     0.0,
            "venue_bowl_wickets":     0,
            "venue_matches":          0,
            "venue_form_score":       50.0,
            # Weather
            "dew_condition_bat_sr":   dew_bat_sr,
            "dew_condition_bowl_eco": dew_bowl_eco,
            "high_humidity_bowl_eco": humid_bowl_eco,
            "night_match_bat_sr":     night_bat_sr,
        }
        rows.append(overall_row)

        # ── Per-venue rows ────────────────────────────────────────────────
        player_venues = bbb.loc[
            bbb["match_id"].isin(last10) & appear_mask, "venue"
        ].dropna().unique()

        for venue in player_venues:
            v_match_ids = list(bbb.loc[
                bbb["match_id"].isin(last10) & (bbb["venue"] == venue), "match_id"
            ].unique())
            v_n = len(v_match_ids)

            vb = _bat_stats( bbb[bbb["match_id"].isin(v_match_ids) & bat_mask])
            vw = _bowl_stats(bbb[bbb["match_id"].isin(v_match_ids) & bowl_mask])

            v_bat_avg  = vb.get("bat_avg",      0.0)
            v_bat_sr   = vb.get("bat_sr",        0.0)
            v_bowl_eco = vw.get("bowl_economy",  0.0)
            v_bowl_wkt = vw.get("bowl_wickets",  0)
            v_form     = _venue_form_score(
                v_bat_avg, v_bat_sr, v_bowl_eco, v_n,
                has_bat=bool(vb), has_bowl=bool(vw),
            )

            rows.append({
                "player_name":            player,
                "reference_date":         ref_date,
                "last_10_matches":        n_matches,
                "venue":                  venue,
                # Overall batting/bowling same as overall row
                "bat_runs_total":         b_stats.get("bat_runs_total", 0),
                "bat_innings":            bat_innings,
                "bat_avg":                bat_avg_val,
                "bat_sr":                 bat_sr_val,
                "bat_boundary_pct":       bat_bpct,
                "bat_dot_pct":            bat_dpct,
                "bat_form_score":         bat_form,
                "bat_trend":              bat_t,
                "bowl_overs":             bowl_overs_val,
                "bowl_wickets":           bowl_wkts_val,
                "bowl_economy":           bowl_eco_val,
                "bowl_dot_pct":           bowl_dpct_val,
                "bowl_avg":               w_stats.get("bowl_avg", 999.0),
                "bowl_sr":                w_stats.get("bowl_sr",  999.0),
                "bowl_form_score":        bowl_form,
                "bowl_trend":             bowl_t,
                # Venue-specific
                "venue_bat_avg":          round(v_bat_avg, 2),
                "venue_bat_sr":           round(v_bat_sr, 2),
                "venue_bowl_economy":     round(v_bowl_eco, 2),
                "venue_bowl_wickets":     v_bowl_wkt,
                "venue_matches":          v_n,
                "venue_form_score":       v_form,
                # Weather
                "dew_condition_bat_sr":   dew_bat_sr,
                "dew_condition_bowl_eco": dew_bowl_eco,
                "high_humidity_bowl_eco": humid_bowl_eco,
                "night_match_bat_sr":     night_bat_sr,
            })

    if not rows:
        raise RuntimeError("No form data computed — ball_by_ball.parquet may be empty")

    out = pd.DataFrame(rows)
    out.to_parquet(OUTPUT_PATH, index=False)

    if verbose:
        overall = out[out["venue"] == ""]
        n_p     = len(overall)
        in_form = int((overall["bat_form_score"] >= 70).sum())
        oof     = int((overall["bat_form_score"] <  35).sum())
        print(f"  Players processed  : {n_p}")
        print(f"  In form  (score>=70): {in_form}")
        print(f"  Out of form (<35)  : {oof}")

        top5 = overall.nlargest(5, "bat_form_score")[["player_name", "bat_form_score", "bat_trend"]]
        print("\n  Top 5 batters by form:")
        for _, r in top5.iterrows():
            print(f"    {r.player_name:<30}  bat_score={r.bat_form_score:.1f}  trend={r.bat_trend}")

        bot5 = overall.nsmallest(5, "bat_form_score")[["player_name", "bat_form_score", "bat_trend"]]
        print("\n  Bottom 5 batters by form:")
        for _, r in bot5.iterrows():
            print(f"    {r.player_name:<30}  bat_score={r.bat_form_score:.1f}  trend={r.bat_trend}")

        top5b = overall.nlargest(5, "bowl_form_score")[["player_name", "bowl_form_score", "bowl_trend"]]
        print("\n  Top 5 bowlers by form:")
        for _, r in top5b.iterrows():
            print(f"    {r.player_name:<30}  bowl_score={r.bowl_form_score:.1f}  trend={r.bowl_trend}")

        print(f"\n  Output: {OUTPUT_PATH}  ({len(out)} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build recent_form.parquet from ball_by_ball.parquet")
    p.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = p.parse_args()
    sys.path.insert(0, str(PROJ_ROOT))
    run(verbose=not args.quiet)
