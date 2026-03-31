# models/train_batting_scenarios.py
# Builds a batting scenario scoring model from historical PSL data.
#
# Produces batting_scenario_model.pkl — a lookup used by engine/batting_scenarios.py
# to generate scenario cards for the pre-match brief.
#
# The four scenarios are:
#   A — Ideal start      (60+ runs, <=2 wkts after 10 overs)
#   B — Tough start      (<=30 runs, 3+ wkts by over 6 — collapse situation)
#   C — Death chase      (innings 2, over 15+, RRR >= 12)
#   D — Conservative     (low-scoring match, team total < 130, wickets in hand)
#
# For each scenario we compute per-batter stats from deliveries matching that
# situation, then score each player 0-100 relative to peers.
#
# Output: models/saved/batting_scenario_model.pkl
#
# Run:  python models/train_batting_scenarios.py

from __future__ import annotations

import csv
import pickle
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
MODELS_DIR    = PROJ_ROOT / "models" / "saved"
OUTPUT_FILE   = MODELS_DIR / "batting_scenario_model.pkl"

BBB_PATH     = PROCESSED_DIR / "ball_by_ball.parquet"
STATS_PATH   = PROCESSED_DIR / "player_stats.parquet"
PLAYER_INDEX = PROCESSED_DIR / "player_index.csv"
PLAYER_INDEX_FALLBACK = PROJ_ROOT.parent / "player_index.csv"

# Minimum balls faced in a scenario to be included in that scenario's ranking
MIN_BALLS_SCENARIO = 20

# Scenario labels
SCENARIOS = ["A", "B", "C", "D"]
SCENARIO_NAMES = {
    "A": "Ideal Start",
    "B": "Tough Start (Collapse)",
    "C": "Death Chase",
    "D": "Conservative Build",
}
SCENARIO_DESC = {
    "A": "60+ at 10 overs, <=2 wickets — build from platform",
    "B": "<=30 at powerplay end, 3+ wickets — reset the innings",
    "C": "Innings 2, over 15+, need 12+ per over — power hitting",
    "D": "Low-scoring game (<130 expected total) — wickets in hand",
}


# ---------------------------------------------------------------------------
# PLAYER META LOADER
# ---------------------------------------------------------------------------

def _load_player_meta(pi_path: Path) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    if not pi_path.exists():
        return meta
    with open(pi_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("player_name", "").strip()
            if not name:
                continue
            style = (row.get("bowling_style") or "").lower()
            meta[name] = {
                "primary_role":    row.get("primary_role", "Batsman").strip(),
                "batting_style":   row.get("batting_style", "Right-hand bat").strip(),
                "is_overseas":     row.get("is_overseas", "False").strip().lower() == "true",
                "is_spin_bowler":  any(w in style for w in ("spin","off","leg","googly","chinaman","slow")),
                "is_pace_bowler":  any(w in style for w in ("fast","medium","seam","swing","pace")),
            }
    return meta


# ---------------------------------------------------------------------------
# SCENARIO FILTERS  (applied to ball-by-ball data)
# ---------------------------------------------------------------------------

def _filter_scenario_A(bbb: pd.DataFrame) -> pd.DataFrame:
    """
    Ideal start: at time of delivery, innings 1, over >= 10,
    innings_score >= 60, innings_wickets <= 2.
    """
    return bbb[
        (bbb["innings"] == 1)
        & (bbb["over"] >= 10)
        & (bbb["innings_score"] >= 60)
        & (bbb["innings_wickets"] <= 2)
    ].copy()


def _filter_scenario_B(bbb: pd.DataFrame) -> pd.DataFrame:
    """
    Tough start / collapse: over <= 8, innings_wickets >= 3, innings_score <= 35.
    Batter arrived in a crisis.
    """
    return bbb[
        (bbb["over"] <= 8)
        & (bbb["innings_wickets"] >= 3)
        & (bbb["innings_score"] <= 35)
    ].copy()


def _filter_scenario_C(bbb: pd.DataFrame) -> pd.DataFrame:
    """
    Death chase: innings 2, over >= 15, rrr >= 12.
    """
    return bbb[
        (bbb["innings"] == 2)
        & (bbb["over"] >= 15)
        & (bbb["rrr"] >= 12.0)
    ].copy()


def _filter_scenario_D(bbb: pd.DataFrame) -> pd.DataFrame:
    """
    Conservative build: low scoring (innings 1 total <= 130 for the match),
    over <= 15, wickets <= 3.
    We approximate by: over <= 15, innings_score < 100, innings_wickets <= 3.
    """
    return bbb[
        (bbb["innings"] == 1)
        & (bbb["over"] <= 15)
        & (bbb["innings_score"] < 100)
        & (bbb["innings_wickets"] <= 3)
        & (bbb["crr"] < 8.0)   # low run rate = conservative match
    ].copy()


SCENARIO_FILTERS = {
    "A": _filter_scenario_A,
    "B": _filter_scenario_B,
    "C": _filter_scenario_C,
    "D": _filter_scenario_D,
}


# ---------------------------------------------------------------------------
# COMPUTE SCENARIO STATS PER BATTER
# ---------------------------------------------------------------------------

def _compute_batter_scenario_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ball_by_ball slice to per-batter stats.
    Returns DataFrame: player_name, balls, runs, dismissals, avg, sr.
    """
    if df.empty:
        return pd.DataFrame(columns=["player_name", "balls", "runs", "dismissals", "avg", "sr"])

    grp = (
        df.groupby("batter", observed=True)
        .agg(
            balls       = ("is_wide",    lambda x: (~x).sum()),
            runs        = ("runs_batter","sum"),
            dismissals  = ("is_wicket",  "sum"),
            fours       = ("runs_batter",lambda x: (x == 4).sum()),
            sixes       = ("runs_batter",lambda x: (x == 6).sum()),
        )
        .reset_index()
        .rename(columns={"batter": "player_name"})
    )
    grp = grp[grp["balls"] >= MIN_BALLS_SCENARIO]
    grp["avg"] = np.where(
        grp["dismissals"] > 0,
        grp["runs"] / grp["dismissals"],
        grp["runs"],   # not out: use total runs
    ).round(1)
    grp["sr"] = (grp["runs"] / grp["balls"] * 100).round(1)
    grp["boundary_pct"] = ((grp["fours"] + grp["sixes"]) / grp["balls"] * 100).round(1)
    return grp


# ---------------------------------------------------------------------------
# SCORE NORMALISER  (0-100 relative score within scenario)
# ---------------------------------------------------------------------------

def _normalise_scores(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """
    Compute a composite 0-100 score for each player within a scenario.
    Weights differ per scenario to reward the relevant skill.
    """
    if df.empty:
        return df

    weights = {
        "A": {"avg": 0.40, "sr": 0.40, "boundary_pct": 0.20},   # balance
        "B": {"avg": 0.55, "sr": 0.30, "boundary_pct": 0.15},   # survival > attack
        "C": {"avg": 0.20, "sr": 0.55, "boundary_pct": 0.25},   # pure aggression
        "D": {"avg": 0.60, "sr": 0.25, "boundary_pct": 0.15},   # consolidation
    }
    w = weights[scenario]

    df = df.copy()
    for col in ["avg", "sr", "boundary_pct"]:
        mn, mx = df[col].min(), df[col].max()
        if mx > mn:
            df[f"{col}_norm"] = (df[col] - mn) / (mx - mn) * 100
        else:
            df[f"{col}_norm"] = 50.0

    df["scenario_score"] = (
        df["avg_norm"]          * w["avg"]
        + df["sr_norm"]          * w["sr"]
        + df["boundary_pct_norm"]* w["boundary_pct"]
    ).round(1)

    # Reliability factor: more balls faced = more reliable
    df["reliability"] = np.clip(df["balls"] / 100, 0.5, 1.0)
    df["scenario_score"] = (df["scenario_score"] * df["reliability"]).round(1)

    return df.sort_values("scenario_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# BUILD SCENARIO LOOKUP
# ---------------------------------------------------------------------------

def _build_scenario_lookup(
    bbb:          pd.DataFrame,
    player_meta:  dict[str, dict],
) -> dict:
    """
    Returns:
        {
            "A": DataFrame(player_name, balls, runs, avg, sr, scenario_score, ...),
            "B": ...,
            "C": ...,
            "D": ...,
        }
    """
    lookup = {}
    for sc, filt_fn in SCENARIO_FILTERS.items():
        filtered = filt_fn(bbb)
        stats    = _compute_batter_scenario_stats(filtered)
        scored   = _normalise_scores(stats, sc)

        # Attach player meta
        scored["primary_role"]   = scored["player_name"].map(lambda p: player_meta.get(p, {}).get("primary_role",  "Batsman"))
        scored["batting_style"]  = scored["player_name"].map(lambda p: player_meta.get(p, {}).get("batting_style", "Right-hand bat"))
        scored["is_overseas"]    = scored["player_name"].map(lambda p: player_meta.get(p, {}).get("is_overseas",   False))

        lookup[sc] = scored

    return lookup


# ---------------------------------------------------------------------------
# BUILD CAREER BASELINE (for players with no scenario data)
# ---------------------------------------------------------------------------

def _build_career_baseline(
    stats:       pd.DataFrame,
    player_meta: dict[str, dict],
) -> dict[str, dict]:
    """
    For players NOT in a scenario's data, fall back to career overall stats.
    Returns {player_name: {"bat_avg", "bat_sr", "bat_boundary_pct", ...}}.
    """
    career = stats[(stats["season"] == 0) & (stats["phase"] == "overall")].copy()
    baseline: dict[str, dict] = {}
    for _, row in career.iterrows():
        p = row["player_name"]
        baseline[p] = {
            "bat_avg":          float(row.get("bat_avg") or 15.0),
            "bat_sr":           float(row.get("bat_sr") or 100.0),
            "bat_boundary_pct": float(row.get("bat_boundary_pct") or 35.0),
            "bat_pp_sr":        0.0,
            "bat_death_sr":     0.0,
            "primary_role":     player_meta.get(p, {}).get("primary_role",  "Batsman"),
            "batting_style":    player_meta.get(p, {}).get("batting_style", "Right-hand bat"),
            "is_overseas":      player_meta.get(p, {}).get("is_overseas",   False),
        }
    return baseline


# ---------------------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------------------

def train(
    bbb_path:    Optional[Path] = None,
    stats_path:  Optional[Path] = None,
    output_path: Optional[Path] = None,
    verbose:     bool = True,
) -> dict:
    """
    Build and save the batting scenario model.

    Returns:
        dict with keys: scenario_lookup, career_baseline, player_meta, scenario_names, scenario_desc
    """
    t0 = time.time()

    bbb_path    = Path(bbb_path)    if bbb_path    else BBB_PATH
    stats_path  = Path(stats_path)  if stats_path  else STATS_PATH
    output_path = Path(output_path) if output_path else OUTPUT_FILE

    if verbose:
        print("[train_batting_scenarios] Loading data...")

    bbb   = pd.read_parquet(bbb_path)
    stats = pd.read_parquet(stats_path)

    pi_path = PLAYER_INDEX if PLAYER_INDEX.exists() else PLAYER_INDEX_FALLBACK
    player_meta = _load_player_meta(pi_path) if pi_path.exists() else {}

    if verbose:
        print(f"[train_batting_scenarios] {len(bbb):,} deliveries, {len(player_meta)} player meta entries")

    # Build scenario scoring tables
    if verbose:
        print("[train_batting_scenarios] Building scenario stats...")

    scenario_lookup = _build_scenario_lookup(bbb, player_meta)

    for sc, df in scenario_lookup.items():
        if verbose:
            print(f"  Scenario {sc} ({SCENARIO_NAMES[sc]}): "
                  f"{len(df)} batters qualifying (>= {MIN_BALLS_SCENARIO} balls)")

    # Career baseline for fallback
    career_baseline = _build_career_baseline(stats, player_meta)

    payload = {
        "scenario_lookup":  scenario_lookup,   # {sc -> DataFrame}
        "career_baseline":  career_baseline,   # {player -> career stats dict}
        "player_meta":      player_meta,
        "scenario_names":   SCENARIO_NAMES,
        "scenario_desc":    SCENARIO_DESC,
        "scenarios":        SCENARIOS,
        "min_balls":        MIN_BALLS_SCENARIO,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    elapsed = time.time() - t0
    size_kb = output_path.stat().st_size / 1024
    if verbose:
        print(f"[train_batting_scenarios] Saved -> {output_path}  ({size_kb:.0f} KB)  [{elapsed:.1f}s]")

    return payload


# ---------------------------------------------------------------------------
# LOOKUP HELPER  (imported by engine/batting_scenarios.py)
# ---------------------------------------------------------------------------

def load_model(model_path: Optional[Path] = None) -> dict:
    model_path = Path(model_path) if model_path else OUTPUT_FILE
    with open(model_path, "rb") as f:
        return pickle.load(f)


def get_scenario_score(
    player:   str,
    scenario: str,
    payload:  dict,
) -> float:
    """
    Return a player's scenario score (0-100).
    Falls back to a career-based proxy if no scenario data.
    """
    sc_df = payload["scenario_lookup"].get(scenario)
    if sc_df is not None:
        row = sc_df[sc_df["player_name"] == player]
        if not row.empty:
            return float(row.iloc[0]["scenario_score"])

    # Fallback: derive from career baseline
    baseline = payload["career_baseline"].get(player, {})
    avg = baseline.get("bat_avg",  15.0)
    sr  = baseline.get("bat_sr",  100.0)

    weights = {"A": (0.4, 0.4), "B": (0.55, 0.3), "C": (0.2, 0.55), "D": (0.6, 0.25)}
    wa, ws  = weights.get(scenario, (0.4, 0.4))

    # Normalise against rough T20 baselines (avg 25, sr 130)
    avg_norm = min(100, avg / 40 * 100)
    sr_norm  = min(100, sr  / 160 * 100)
    return round(avg_norm * wa + sr_norm * ws, 1)


def rank_players_for_scenario(
    players:  list[str],
    scenario: str,
    payload:  dict,
) -> list[tuple[str, float]]:
    """
    Rank a list of players for a scenario.
    Returns [(player_name, score), ...] sorted highest first.
    """
    scored = [(p, get_scenario_score(p, scenario, payload)) for p in players]
    return sorted(scored, key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    payload = train(verbose=True)

    print(f"\n{'='*60}")
    print(f"  batting_scenario_model.pkl -- top performers per scenario")
    print(f"{'='*60}")

    for sc in SCENARIOS:
        df = payload["scenario_lookup"][sc]
        print(f"\n  Scenario {sc}: {SCENARIO_NAMES[sc]}")
        print(f"  Context : {SCENARIO_DESC[sc]}")
        print(f"  {'Player':<28}  Balls  Runs  Avg   SR    Score")
        print(f"  {'-'*65}")
        for _, r in df.head(8).iterrows():
            print(
                f"  {r['player_name']:<28}  {int(r['balls']):>5}  "
                f"{int(r['runs']):>4}  {r['avg']:>5.1f}  {r['sr']:>5.1f}  "
                f"{r['scenario_score']:>5.1f}"
            )

    # Demo: rank Lahore 2026 squad for a death chase
    lahore_squad = [
        "Fakhar Zaman", "Abdullah Shafique", "Sikandar Raza",
        "Shaheen Shah Afridi", "Hussain Talat", "Tayyab Tahir",
    ]
    print(f"\n  Demo: rank sample squad for death chase (Scenario C):")
    ranked = rank_players_for_scenario(lahore_squad, "C", payload)
    for p, sc_score in ranked:
        print(f"    {p:<30}  {sc_score:.1f}")

    print(f"{'='*60}\n")
