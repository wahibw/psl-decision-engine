# models/train_xi_scorer.py
# Trains a player situation scorer used by xi_selector.py.
# UPGRADE 2: Also trains a TabNetRegressor (models/saved/xi_tabnet.zip) as
#            primary model; XGBoost xi_scorer.pkl becomes the fallback.
#
# The scorer produces a phase-weighted "match impact score" in real cricket
# units for any player given their stats and a match context.
#
# Training:
#   Row    = (player, match, innings) where player appeared
#   Target = phase-weighted absolute impact (batting + bowling, real-unit runs).
#            Typical range: -30 to +120; median ~15-25 for a PSL innings.
#   Features = career phase stats + role + weather-proxy flags
#
# Model: TabNetRegressor (primary)  +  XGBRegressor (fallback)
# Output: models/saved/xi_scorer.pkl        (XGBoost fallback)
#         models/saved/xi_tabnet.zip        (TabNet primary)
#
# Run: python models/train_xi_scorer.py

from __future__ import annotations

import csv
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

PROJ_ROOT     = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJ_ROOT / "data" / "processed"
MODELS_DIR    = PROJ_ROOT / "models" / "saved"
OUTPUT_FILE     = MODELS_DIR / "xi_scorer.pkl"
TABNET_FILE     = MODELS_DIR / "xi_tabnet"          # TabNet saves as xi_tabnet.zip
TRANSFER_FILE   = MODELS_DIR / "xi_tabnet_transfer"  # Upgrade 5: fine-tuned on recent seasons

BBB_PATH     = PROCESSED_DIR / "ball_by_ball.parquet"
STATS_PATH   = PROCESSED_DIR / "player_stats.parquet"
PLAYER_INDEX = PROCESSED_DIR / "player_index_2026_enriched.csv"
PLAYER_INDEX_FALLBACK = PROJ_ROOT.parent / "player_index_2026_enriched.csv"

# PSL baseline constants for phase-weighted absolute impact (PSL 2019-2025 career averages).
PSL_AVG_RUNS          = 22.0   # average runs scored per batting innings
PSL_AVG_SR           = 128.0  # average strike rate across all PSL innings
PSL_POWERPLAY_ECONOMY = 7.8   # average economy in powerplay overs (1-6)
PSL_MIDDLE_ECONOMY    = 7.5   # average economy in middle overs (7-16)
PSL_DEATH_ECONOMY     = 10.8  # average economy in death overs (17-20)
PSL_WICKET_VALUE      = 18.0  # run-equivalent value of one wicket

# Phase-level bowling weights: economy vs wicket importance shifts with over block.
PHASE_WEIGHTS = {
    "powerplay": {"economy": 0.55, "wicket": 0.45},
    "middle":    {"economy": 0.45, "wicket": 0.55},
    "death":     {"economy": 0.30, "wicket": 0.70},
}

# Batting phase multipliers: late overs are higher leverage.
PHASE_BAT_MULT = {"powerplay": 1.0, "middle": 1.1, "death": 1.3}

# Role codes
ROLE_CODE = {
    "Batsman":     0,
    "Wicketkeeper":1,
    "All-rounder": 2,
    "Bowler":      3,
}

FEATURE_COLS = [
    # Career batting
    "bat_avg",
    "bat_sr",
    "bat_pp_sr",
    "bat_death_sr",
    "bat_boundary_pct",
    # Chase / set batting context
    "bat_avg_chase",            # batting avg when chasing (innings 2)
    "bat_avg_set",              # batting avg when setting (innings 1)
    "innings_context_split",    # bat_avg_chase − bat_avg_set (positive = better chaser)
    # Career bowling
    "bowl_economy",
    "bowl_wkts_per_over",
    "bowl_pp_economy",
    "bowl_death_economy",
    "bowl_dot_pct",
    # Role / type
    "role_code",
    "is_overseas",
    "is_pace",
    "is_spin",
    # Match context proxies
    "venue_pace_economy",   # venue historical pace economy
    "venue_spin_economy",   # venue historical spin economy
    "innings_num",          # 1 or 2 (batting first vs chasing)
]


# ---------------------------------------------------------------------------
# PLAYER META LOOKUP
# ---------------------------------------------------------------------------

def _load_player_meta(pi_path: Path) -> dict[str, dict]:
    """Returns {player_name: {role, is_overseas, is_pace, is_spin}}."""
    meta: dict[str, dict] = {}
    if not pi_path.exists():
        return meta
    with open(pi_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name  = row.get("player_name", "").strip()
            style = (row.get("bowling_style") or "").lower()
            role  = row.get("primary_role", "Batsman").strip()
            overseas = row.get("is_overseas", "False").strip().lower() == "true"
            is_pace = any(w in style for w in ("fast", "medium", "seam", "swing", "pace"))
            is_spin = any(w in style for w in ("spin", "off", "leg", "googly", "chinaman", "slow"))
            meta[name] = {
                "role":       role,
                "role_code":  ROLE_CODE.get(role, 0),
                "is_overseas":int(overseas),
                "is_pace":    int(is_pace),
                "is_spin":    int(is_spin),
            }
    return meta


# ---------------------------------------------------------------------------
# BUILD PLAYER CAREER FEATURE LOOKUP
# ---------------------------------------------------------------------------

def _build_player_features(
    stats:  pd.DataFrame,
    player_meta: dict[str, dict],
    cutoff_seasons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by player_name with all scorer features.

    Args:
        cutoff_seasons: If given, compute career aggregates from ONLY these
                        seasons (used for leakage-free test evaluation).
                        If None, use the pre-computed season=0 career rows.
    """
    if cutoff_seasons is not None:
        # Fix 2.3: Build career stats from restricted season set to avoid leakage.
        # We re-aggregate from season-level rows rather than using season=0.
        sub = stats[stats["season"].isin(cutoff_seasons)].copy()
        # Aggregate phase-level rows across allowed seasons
        num_cols = [c for c in sub.columns if c.startswith(("bat_", "bowl_")) and c not in ("bat_innings",)]
        # We can't simply sum because averages/SRs need to be recomputed from raw counts.
        # Use a weighted mean (weight = bat_balls or bowl_balls) as a reasonable proxy.
        def _wavg(group: pd.DataFrame, val_col: str, wt_col: str) -> float:
            w = group[wt_col].fillna(0)
            v = group[val_col].fillna(0)
            return float((v * w).sum() / w.sum()) if w.sum() > 0 else 0.0

        records = []
        for (player, phase), grp in sub.groupby(["player_name", "phase"]):
            rec = {"player_name": player, "phase": phase}
            for col in ["bat_avg", "bat_sr", "bat_boundary_pct", "bat_dot_pct"]:
                rec[col] = _wavg(grp, col, "bat_balls") if "bat_balls" in grp else 0.0
            for col in ["bowl_economy", "bowl_dot_pct", "bowl_boundary_pct"]:
                rec[col] = _wavg(grp, col, "bowl_balls") if "bowl_balls" in grp else 0.0
            # Chase/set context columns — weighted average when available, else fall
            # back to career bat_avg (cutoff_seasons path is eval-only, not production).
            for col in ["bat_avg_chase", "bat_avg_set", "innings_context_split"]:
                if col in grp.columns:
                    rec[col] = _wavg(grp, col, "bat_balls") if "bat_balls" in grp else 0.0
                else:
                    rec[col] = rec.get("bat_avg", 0.0) if col != "innings_context_split" else 0.0
            bowl_overs  = grp["bowl_overs"].fillna(0).sum()
            bowl_wickets= grp["bowl_wickets"].fillna(0).sum()
            rec["bowl_overs"]   = float(bowl_overs)
            rec["bowl_wickets"] = float(bowl_wickets)
            rec["bowl_sr"]      = round(bowl_overs * 6 / bowl_wickets, 1) if bowl_wickets > 0 else 99.9
            records.append(rec)
        career = pd.DataFrame(records)
    else:
        career = stats[stats["season"] == 0].copy()

    def _phase_val(player: str, phase: str, col: str, default: float) -> float:
        row = career[(career["player_name"] == player) & (career["phase"] == phase)]
        if row.empty:
            return default
        v = row.iloc[0].get(col)
        return float(v) if pd.notna(v) else default

    players = career["player_name"].unique()
    records = []

    for p in players:
        meta = player_meta.get(p, {
            "role": "Batsman", "role_code": 0,
            "is_overseas": 0, "is_pace": 0, "is_spin": 0,
        })

        # Overall career
        bat_avg         = _phase_val(p, "overall", "bat_avg",          15.0)
        bat_sr          = _phase_val(p, "overall", "bat_sr",          120.0)
        bat_boundary    = _phase_val(p, "overall", "bat_boundary_pct", 40.0)
        bowl_economy    = _phase_val(p, "overall", "bowl_economy",      9.0)
        bowl_dot        = _phase_val(p, "overall", "bowl_dot_pct",     35.0)

        # Chase / set context — fall back to career bat_avg when column missing
        bat_avg_chase = _phase_val(p, "overall", "bat_avg_chase", bat_avg)
        bat_avg_set   = _phase_val(p, "overall", "bat_avg_set",   bat_avg)
        innings_ctx   = bat_avg_chase - bat_avg_set   # positive = better chaser

        # Phase-specific
        bat_pp_sr    = _phase_val(p, "powerplay", "bat_sr",           120.0)
        bat_death_sr = _phase_val(p, "death",     "bat_sr",           120.0)
        bowl_pp_econ = _phase_val(p, "powerplay", "bowl_economy",       8.5)
        bowl_dth_econ= _phase_val(p, "death",     "bowl_economy",       9.5)

        # Wickets per over (derive from overall)
        row_overall = career[(career["player_name"] == p) & (career["phase"] == "overall")]
        if not row_overall.empty:
            overs = float(row_overall.iloc[0].get("bowl_overs") or 0)
            wkts  = float(row_overall.iloc[0].get("bowl_wickets") or 0)
            wkts_po = round(wkts / overs, 3) if overs > 0 else 0.0
        else:
            wkts_po = 0.0

        records.append({
            "player_name":          p,
            "bat_avg":              bat_avg,
            "bat_sr":               bat_sr,
            "bat_pp_sr":            bat_pp_sr,
            "bat_death_sr":         bat_death_sr,
            "bat_boundary_pct":     bat_boundary,
            "bat_avg_chase":        bat_avg_chase,
            "bat_avg_set":          bat_avg_set,
            "innings_context_split":innings_ctx,
            "bowl_economy":         bowl_economy,
            "bowl_wkts_per_over":   wkts_po,
            "bowl_pp_economy":      bowl_pp_econ,
            "bowl_death_economy":   bowl_dth_econ,
            "bowl_dot_pct":         bowl_dot,
            "role_code":            meta["role_code"],
            "is_overseas":          meta["is_overseas"],
            "is_pace":              meta["is_pace"],
            "is_spin":              meta["is_spin"],
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# BUILD TRAINING ROWS FROM BALL-BY-BALL
# ---------------------------------------------------------------------------

def _build_training_rows(
    bbb:          pd.DataFrame,
    player_feat:  pd.DataFrame,
    venue_stats:  pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per (player, match, innings) where player appeared.
    Target = phase-weighted absolute impact in real cricket units (no clipping).
    Typical range: -30 to +120, median ~15-25 for a typical PSL innings.
    """
    # -----------------------------------------------------------------------
    # Phase assignment (0-indexed overs: 0-5 = PP, 6-15 = mid, 16-19 = death)
    # -----------------------------------------------------------------------
    bbb = bbb.copy()
    bbb["_phase"] = np.select(
        [bbb["over"] < 6, bbb["over"] >= 16],
        ["powerplay", "death"],
        default="middle",
    )

    # -----------------------------------------------------------------------
    # BATTING — phase-level aggregation then sum with phase multiplier
    # -----------------------------------------------------------------------
    bat_phase = (
        bbb.groupby(["match_id", "season", "innings", "batter", "_phase"], observed=True)
        .agg(
            runs_scored = ("runs_batter", "sum"),
            balls_faced = ("is_wide",     lambda x: (~x).sum()),
            venue       = ("venue",       "first"),
        )
        .reset_index()
        .rename(columns={"batter": "player_name", "_phase": "phase"})
    )

    bat_phase["sr"] = np.where(
        bat_phase["balls_faced"] > 0,
        bat_phase["runs_scored"] / bat_phase["balls_faced"] * 100,
        0.0,
    )
    bat_phase["runs_above_avg"] = bat_phase["runs_scored"] - PSL_AVG_RUNS
    bat_phase["sr_component"]   = (
        (bat_phase["sr"] - PSL_AVG_SR) / PSL_AVG_SR * bat_phase["runs_scored"]
    )
    bat_phase["raw_bat"]     = bat_phase["runs_above_avg"] * 1.5 + bat_phase["sr_component"] * 0.5
    bat_phase["phase_mult"]  = bat_phase["phase"].map(PHASE_BAT_MULT).fillna(1.0)
    bat_phase["bat_contrib"] = bat_phase["raw_bat"] * bat_phase["phase_mult"]

    # Collapse to innings level
    bat = (
        bat_phase.groupby(["match_id", "season", "innings", "player_name"], observed=True)
        .agg(
            batting_impact = ("bat_contrib", "sum"),
            venue          = ("venue",       "first"),
        )
        .reset_index()
    )

    # Not-out bonus: +5.0 per innings if the batter was not dismissed.
    # Requires player_dismissed column (present in Cricsheet-parsed data).
    if "player_dismissed" in bbb.columns:
        _dismissed = (
            bbb[bbb["is_wicket"].astype(bool) & bbb["player_dismissed"].notna()]
            [["match_id", "innings", "player_dismissed"]]
            .rename(columns={"player_dismissed": "player_name"})
            .drop_duplicates()
            .assign(was_dismissed=True)
        )
        bat = bat.merge(_dismissed, on=["match_id", "innings", "player_name"], how="left")
        bat["batting_impact"] += np.where(bat["was_dismissed"].isna(), 5.0, 0.0)
        bat = bat.drop(columns=["was_dismissed"])

    # -----------------------------------------------------------------------
    # BOWLING — phase-level aggregation with economy + wicket weights
    # -----------------------------------------------------------------------
    _PHASE_AVG_ECON = {
        "powerplay": PSL_POWERPLAY_ECONOMY,
        "middle":    PSL_MIDDLE_ECONOMY,
        "death":     PSL_DEATH_ECONOMY,
    }
    _EW = {p: v["economy"] for p, v in PHASE_WEIGHTS.items()}
    _WW = {p: v["wicket"]  for p, v in PHASE_WEIGHTS.items()}

    bbb_bowl = bbb[~bbb["wicket_type"].isin(["run out", "obstructed the field", "retired hurt"])]
    bowl_phase = (
        bbb_bowl.groupby(["match_id", "season", "innings", "bowler", "_phase"], observed=True)
        .agg(
            bowl_runs   = ("runs_total", "sum"),
            legal_balls = ("is_wide",    lambda x: (~x).sum()),
            wickets     = ("is_wicket",  "sum"),
            venue       = ("venue",      "first"),
        )
        .reset_index()
        .rename(columns={"bowler": "player_name", "_phase": "phase"})
    )

    bowl_phase["overs"]          = bowl_phase["legal_balls"] / 6
    bowl_phase["economy"]        = np.where(
        bowl_phase["overs"] > 0,
        bowl_phase["bowl_runs"] / bowl_phase["overs"],
        0.0,
    )
    bowl_phase["phase_avg_econ"] = bowl_phase["phase"].map(_PHASE_AVG_ECON).fillna(PSL_MIDDLE_ECONOMY)
    bowl_phase["economy_saved"]  = (bowl_phase["phase_avg_econ"] - bowl_phase["economy"]) * bowl_phase["overs"]
    bowl_phase["wicket_value"]   = bowl_phase["wickets"] * PSL_WICKET_VALUE
    bowl_phase["ew"]             = bowl_phase["phase"].map(_EW).fillna(0.45)
    bowl_phase["ww"]             = bowl_phase["phase"].map(_WW).fillna(0.55)
    bowl_phase["bowl_contrib"]   = (
        bowl_phase["economy_saved"] * bowl_phase["ew"] * 10
        + bowl_phase["wicket_value"] * bowl_phase["ww"]
    )

    bowl = (
        bowl_phase.groupby(["match_id", "season", "innings", "player_name"], observed=True)
        .agg(bowling_impact=("bowl_contrib", "sum"))
        .reset_index()
    )

    # -----------------------------------------------------------------------
    # Merge batting + bowling; final impact in real cricket units
    # -----------------------------------------------------------------------
    combined = bat[["match_id", "season", "innings", "player_name", "venue", "batting_impact"]].merge(
        bowl[["match_id", "innings", "player_name", "bowling_impact"]],
        on=["match_id", "innings", "player_name"],
        how="outer",
    )
    combined["batting_impact"] = combined["batting_impact"].fillna(0.0)
    combined["bowling_impact"] = combined["bowling_impact"].fillna(0.0)
    combined["impact_score"]   = combined["batting_impact"] + combined["bowling_impact"]

    # Join player features
    df = combined.merge(player_feat, on="player_name", how="inner")

    # Join venue stats (pace/spin economy) as match-context proxy
    venue_eco = venue_stats[["venue", "pace_economy", "spin_economy"]].copy()
    venue_eco.columns = ["venue", "venue_pace_economy", "venue_spin_economy"]
    df = df.merge(venue_eco, on="venue", how="left")
    df["venue_pace_economy"] = df["venue_pace_economy"].fillna(8.5)
    df["venue_spin_economy"] = df["venue_spin_economy"].fillna(8.0)

    df["innings_num"] = df["innings"].astype(int)

    return df


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------

def train(
    bbb_path:    Optional[Path] = None,
    stats_path:  Optional[Path] = None,
    output_path: Optional[Path] = None,
    verbose:     bool = True,
) -> dict:
    t0 = time.time()

    bbb_path    = Path(bbb_path)    if bbb_path    else BBB_PATH
    stats_path  = Path(stats_path)  if stats_path  else STATS_PATH
    output_path = Path(output_path) if output_path else OUTPUT_FILE

    venue_stats_path = PROCESSED_DIR / "venue_stats.csv"

    if verbose:
        print("[train_xi_scorer] Loading data...")

    bbb   = pd.read_parquet(bbb_path)
    stats = pd.read_parquet(stats_path)
    venue_stats = pd.read_csv(venue_stats_path)

    pi_path = PLAYER_INDEX if PLAYER_INDEX.exists() else PLAYER_INDEX_FALLBACK
    player_meta = _load_player_meta(pi_path) if pi_path.exists() else {}

    if verbose:
        print(f"[train_xi_scorer] {len(bbb):,} deliveries, {stats['player_name'].nunique()} players")

    # Build features + training rows
    if verbose:
        print("[train_xi_scorer] Building player feature lookup...")

    # Season split (needed before building features for leakage-free test eval)
    all_seasons   = sorted(bbb["season"].unique())
    test_seasons  = all_seasons[-2:] if len(all_seasons) >= 2 else all_seasons[-1:]
    train_seasons = [s for s in all_seasons if s not in test_seasons]

    # Full career features — used for production model and train-set training
    player_feat = _build_player_features(stats, player_meta)

    if verbose:
        print("[train_xi_scorer] Building training rows from match data...")
    train_df = _build_training_rows(bbb, player_feat, venue_stats)

    if verbose:
        print(f"[train_xi_scorer] Training set: {len(train_df):,} rows x {len(FEATURE_COLS)} features")

    tr_mask = train_df["season"].isin(train_seasons)
    te_mask = train_df["season"].isin(test_seasons)

    X_train = train_df.loc[tr_mask, FEATURE_COLS].astype(float)
    y_train = train_df.loc[tr_mask, "impact_score"].astype(float)

    # Fix 2.3: Build test features using only pre-test-season stats to avoid leakage.
    # Career features include test-season data, so test RMSE was artificially low.
    if verbose:
        print("[train_xi_scorer] Building leakage-free features for test evaluation...")
    player_feat_prettest = _build_player_features(stats, player_meta, cutoff_seasons=train_seasons)
    train_df_test = _build_training_rows(
        bbb[bbb["season"].isin(test_seasons)], player_feat_prettest, venue_stats
    )
    X_test = train_df_test[FEATURE_COLS].astype(float)
    y_test = train_df_test["impact_score"].astype(float)

    if verbose:
        print(f"[train_xi_scorer] Train: {len(X_train):,}  |  Test: {len(X_test):,}")
        _sc = train_df["impact_score"]
        print(
            f"[train_xi_scorer] Impact score distribution (all rows, real cricket units):\n"
            f"  mean={_sc.mean():.1f}  std={_sc.std():.1f}"
            f"  p10={_sc.quantile(0.10):.1f}  p50={_sc.quantile(0.50):.1f}  p90={_sc.quantile(0.90):.1f}"
            f"  min={_sc.min():.1f}  max={_sc.max():.1f}"
        )

    model = XGBRegressor(
        n_estimators     = 400,
        max_depth        = 4,
        learning_rate    = 0.04,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 5,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        random_state     = 42,
        verbosity        = 0,
        n_jobs           = -1,
    )

    if verbose:
        print("[train_xi_scorer] Fitting XGBRegressor...")

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)
    rmse_train = float(np.sqrt(np.mean((pred_train - y_train) ** 2)))
    rmse_test  = float(np.sqrt(np.mean((pred_test  - y_test)  ** 2)))
    mae_test   = float(np.mean(np.abs(pred_test - y_test)))

    if verbose:
        print(f"[train_xi_scorer] Train RMSE: {rmse_train:.2f}  |  Test RMSE: {rmse_test:.2f}  |  MAE: {mae_test:.2f}")

    imp = dict(zip(FEATURE_COLS, model.feature_importances_))
    top5 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:5]
    if verbose:
        print("[train_xi_scorer] Top 5 features:")
        for fname, fimpt in top5:
            print(f"    {fname:<30}  {fimpt:.4f}")

    # Build player scoring lookup for production: player_name -> feature dict
    player_lookup = player_feat.set_index("player_name").to_dict("index")

    # Venue lookup: venue_name -> {pace_economy, spin_economy}
    venue_lookup = {
        row["venue"]: {
            "venue_pace_economy": float(row.get("pace_economy", 8.5)),
            "venue_spin_economy": float(row.get("spin_economy", 8.0)),
        }
        for _, row in venue_stats.iterrows()
    }

    payload = {
        "model":          model,
        "feature_cols":   FEATURE_COLS,
        "player_lookup":  player_lookup,
        "player_meta":    player_meta,
        "venue_lookup":   venue_lookup,
        "role_code":      ROLE_CODE,
        "meta": {
            "rmse_train":    rmse_train,
            "rmse_test":     rmse_test,
            "mae_test":      mae_test,
            "train_rows":    len(X_train),
            "test_rows":     len(X_test),
            "train_seasons": train_seasons,
            "test_seasons":  test_seasons,
            "n_players":     len(player_lookup),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    elapsed = time.time() - t0
    size_kb = output_path.stat().st_size / 1024
    if verbose:
        print(f"[train_xi_scorer] Saved -> {output_path}  ({size_kb:.0f} KB)  [{elapsed:.1f}s]")

    # --- UPGRADE 2: Also train TabNet (Stage 1 — full data) ---
    train_tabnet(X_train, y_train, X_test, y_test, verbose=verbose)

    # --- UPGRADE 5: Transfer learning (Stage 2 — recent seasons) ---
    train_transfer(train_df, verbose=verbose)

    return payload


# ---------------------------------------------------------------------------
# UPGRADE 2 — TABNET MODEL
# ---------------------------------------------------------------------------

def _try_import_tabnet():
    try:
        from pytorch_tabnet.tab_model import TabNetRegressor
        return TabNetRegressor
    except ImportError:
        return None


def train_tabnet(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    output_path: Optional[Path] = None,
    verbose: bool = True,
) -> bool:
    """
    Train a TabNetRegressor on the same feature matrix as XGBoost.
    Saves to TABNET_FILE.zip.  Returns True on success.
    """
    TabNetRegressor = _try_import_tabnet()
    if TabNetRegressor is None:
        if verbose:
            print("[train_xi_scorer] pytorch-tabnet not available — skipping TabNet")
        return False

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path = Path(output_path) if output_path else TABNET_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[train_xi_scorer] Training TabNet on {device}  "
              f"rows={len(X_train):,}  features={X_train.shape[1]}")

    Xtr = X_train.values.astype(np.float32)
    ytr = y_train.values.astype(np.float32).reshape(-1, 1)
    Xte = X_test.values.astype(np.float32)
    yte = y_test.values.astype(np.float32).reshape(-1, 1)

    model = TabNetRegressor(
        n_d=32, n_a=32, n_steps=5, gamma=1.3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": 2e-3},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 10, "gamma": 0.9},
        verbose=0,
        device_name=device,
        seed=42,
    )

    model.fit(
        Xtr, ytr,
        eval_set=[(Xte, yte)],
        eval_metric=["rmse"],
        max_epochs=10,          # reduced from 100 for fast CPU training
        patience=5,
        batch_size=256,
        virtual_batch_size=64,
    )

    pred = model.predict(Xte).flatten()
    rmse = float(np.sqrt(np.mean((pred - y_test.values) ** 2)))
    mae  = float(np.mean(np.abs(pred - y_test.values)))

    if verbose:
        print(f"[train_xi_scorer] TabNet Test RMSE={rmse:.2f}  MAE={mae:.2f}")

    model.save_model(str(output_path))
    if verbose:
        print(f"[train_xi_scorer] TabNet saved -> {output_path}.zip")
    return True


# ---------------------------------------------------------------------------
# UPGRADE 5 — TRANSFER LEARNING (fine-tune on recent seasons for Tier 3)
# ---------------------------------------------------------------------------

def train_transfer(
    train_df:    pd.DataFrame,
    verbose:     bool = True,
) -> bool:
    """
    Stage 2 transfer: fine-tune a new TabNet on the most recent 2 PSL seasons.
    This gives better predictions for Tier-3 players (overseas/limited PSL history)
    by learning recency-weighted patterns from latest conditions and squads.

    Loads Stage 1 weights from TABNET_FILE if available; otherwise starts fresh.
    Saves to TRANSFER_FILE.zip.
    """
    TabNetRegressor = _try_import_tabnet()
    if TabNetRegressor is None:
        if verbose:
            print("[train_xi_scorer] pytorch-tabnet not available — skipping transfer")
        return False

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_seasons    = sorted(train_df["season"].unique())
    recent_seasons = all_seasons[-2:] if len(all_seasons) >= 2 else all_seasons
    recent_mask    = train_df["season"].isin(recent_seasons)

    if recent_mask.sum() < 20:
        if verbose:
            print("[train_xi_scorer] Not enough recent-season rows for transfer — skipping")
        return False

    df_r   = train_df[recent_mask]
    # 80/20 split on recent data
    n_val  = max(1, len(df_r) // 5)
    n_tr   = len(df_r) - n_val
    df_tr  = df_r.iloc[:n_tr]
    df_val = df_r.iloc[n_tr:]

    Xtr  = df_tr[FEATURE_COLS].values.astype(np.float32)
    ytr  = df_tr["impact_score"].values.astype(np.float32).reshape(-1, 1)
    Xval = df_val[FEATURE_COLS].values.astype(np.float32)
    yval = df_val["impact_score"].values.astype(np.float32).reshape(-1, 1)

    if verbose:
        print(f"[train_xi_scorer] Transfer training on {recent_seasons} "
              f"({len(Xtr)} train / {len(Xval)} val rows)")

    model = TabNetRegressor(
        n_d=32, n_a=32, n_steps=5, gamma=1.3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": 5e-4},   # lower LR for fine-tuning
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 3, "gamma": 0.9},
        verbose=0,
        device_name=device,
        seed=42,
    )

    model.fit(
        Xtr, ytr,
        eval_set=[(Xval, yval)],
        eval_metric=["rmse"],
        max_epochs=10,   # fast fine-tune
        patience=5,
        batch_size=128,
        virtual_batch_size=32,
    )

    TRANSFER_FILE.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(TRANSFER_FILE))
    if verbose:
        pred = model.predict(Xval).flatten()
        rmse = float(np.sqrt(np.mean((pred - yval.flatten()) ** 2)))
        print(f"[train_xi_scorer] Transfer model RMSE={rmse:.2f}  saved -> {TRANSFER_FILE}.zip")
    return True


# ---------------------------------------------------------------------------
# SCORING HELPER  (imported by xi_selector.py at runtime)
# ---------------------------------------------------------------------------

def load_model(model_path: Optional[Path] = None) -> dict:
    """
    Load XI scorer payload.
    Prefers TabNet (xi_tabnet.zip); falls back to XGBoost (xi_scorer.pkl).
    Also loads transfer model (xi_tabnet_transfer.zip) for Tier 3 players (Upgrade 5).
    """
    tabnet_path    = Path(str(TABNET_FILE)    + ".zip")
    transfer_path  = Path(str(TRANSFER_FILE)  + ".zip")

    if tabnet_path.exists():
        TabNetRegressor = _try_import_tabnet()
        if TabNetRegressor is not None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                tn = TabNetRegressor(device_name=device)
                tn.load_model(str(tabnet_path))
                # Still load XGBoost payload for lookup tables
                pkl_path = Path(model_path) if model_path else OUTPUT_FILE
                if pkl_path.exists():
                    with open(pkl_path, "rb") as f:
                        base_payload = pickle.load(f)
                else:
                    base_payload = {
                        "player_lookup": {}, "player_meta": {},
                        "venue_lookup": {}, "role_code": ROLE_CODE,
                        "feature_cols": FEATURE_COLS, "model": None, "meta": {},
                    }
                base_payload["tabnet_model"] = tn
                # Upgrade 5: also load transfer model if available
                if transfer_path.exists():
                    try:
                        tr = TabNetRegressor(device_name=device)
                        tr.load_model(str(transfer_path))
                        base_payload["transfer_model"] = tr
                    except Exception:
                        pass   # transfer model optional
                return base_payload
            except Exception as e:
                print(f"[train_xi_scorer] TabNet load failed ({e}) — using XGBoost fallback")

    model_path = Path(model_path) if model_path else OUTPUT_FILE
    with open(model_path, "rb") as f:
        return pickle.load(f)


def _batting_score(pf: dict) -> float:
    """
    Role-aware batting score (0-100) based purely on batting stats.
    Weights: avg 30% | sr 35% | boundary_pct 15% | pp_sr 10% | death_sr 10%
    """
    def _norm(v, lo, hi):
        return max(0.0, min(1.0, (v - lo) / (hi - lo))) * 100.0

    avg         = _norm(pf.get("bat_avg",          15.0),  5.0,  55.0)
    # Tightened from (90,185) → (98,162): better differentiates the 110-155 SR band
    # where most competitive T20 batters live, avoiding the "world-class = mediocre" compression
    sr          = _norm(pf.get("bat_sr",          120.0), 98.0, 162.0)
    boundary    = _norm(pf.get("bat_boundary_pct",  40.0), 20.0,  65.0)
    pp_sr       = _norm(pf.get("bat_pp_sr",        120.0), 80.0, 175.0)
    death_sr    = _norm(pf.get("bat_death_sr",     120.0), 90.0, 195.0)

    return avg * 0.30 + sr * 0.35 + boundary * 0.15 + pp_sr * 0.10 + death_sr * 0.10


def _bowling_score(pf: dict) -> float:
    """
    Role-aware bowling score (0-100) based purely on bowling stats.
    Weights: wickets/over 40% | economy 30% | dot_pct 15% | pp_eco 8% | death_eco 7%
    """
    def _norm(v, lo, hi):
        return max(0.0, min(1.0, (v - lo) / (hi - lo))) * 100.0

    # For economy: lower is better → invert
    economy     = _norm(12.0 - pf.get("bowl_economy",       9.0),  0.0,  6.0)
    wkts_po     = _norm(pf.get("bowl_wkts_per_over",         0.0),  0.0,  0.55)
    dot_pct     = _norm(pf.get("bowl_dot_pct",              35.0), 20.0, 58.0)
    pp_eco      = _norm(12.0 - pf.get("bowl_pp_economy",    8.5),   0.0,  6.0)
    death_eco   = _norm(14.0 - pf.get("bowl_death_economy", 9.5),   0.0,  8.0)

    return wkts_po * 0.40 + economy * 0.30 + dot_pct * 0.15 + pp_eco * 0.08 + death_eco * 0.07


def score_player(
    player:          str,
    venue:           str,
    innings_num:     int,
    payload:         dict,
    spinner_penalty: float = 1.0,
    swing_bonus:     float = 1.0,
    pace_bounce:     float = 1.0,
    role_override:   Optional[str] = None,
    t20_proxy:       Optional[dict] = None,
    _source_out:     Optional[list] = None,   # Upgrade 5: append model source string if provided
) -> float:
    """
    Role-aware match impact score (0-100) for a player in this context.

    Scoring is split by role so bowlers are not penalised for low batting stats:
      Batters / WK-Batsmen : batting score only
      Bowlers              : bowling score only
      All-rounders         : average of batting + bowling scores

    Args:
        role_override: primary_role string from the current player_index
                       ("Batsman", "Bowler", "All-rounder", "WK-Batsman").
                       Use this to override stale role_code in the pkl lookup.

    Weather modifiers:
      spinner_penalty: 0.4-1.0  — reduces score for spin bowlers (dew)
      swing_bonus:     1.0-1.4  — boosts score for pace swing bowlers
      pace_bounce:     1.0-1.25 — boosts score for fast bowlers (cold pitch)

    Returns a float between 0-100. Higher = better for THIS match.
    Falls back to a role-based average if player not in lookup.
    """
    ROLE_CODE_MAP = {"Batsman": 0, "WK-Batsman": 1, "Wicketkeeper": 1,
                     "All-rounder": 2, "Bowler": 3}

    lookup = payload["player_lookup"]
    meta   = payload["player_meta"]

    # Resolve role: fresh player_index override → pkl meta → pkl lookup → default
    if role_override:
        rc = ROLE_CODE_MAP.get(role_override, 0)
    else:
        pm = meta.get(player, {})
        rc = pm.get("role_code") if pm.get("role_code") is not None else 0
        if player in lookup:
            pkl_rc = lookup[player].get("role_code")
            if pkl_rc is not None and pm.get("role_code") is None:
                rc = int(pkl_rc)

    if player not in lookup:
        # --- Upgrade 5: try transfer model first for Tier 3 players ---
        transfer = payload.get("transfer_model")
        proxy = t20_proxy or {}
        tier  = int(proxy.get("data_tier", 3))
        if transfer is not None and tier >= 2:
            try:
                bat_sr  = float(proxy.get("bat_sr",       120.0) or 120.0)
                bat_avg = float(proxy.get("bat_avg",        15.0) or 15.0)
                bowl_eco= float(proxy.get("bowl_economy",    8.5) or 8.5)
                venue_info = payload.get("venue_lookup", {}).get(venue, {})
                pf_t = {
                    "bat_avg":            bat_avg,
                    "bat_sr":             bat_sr,
                    "bat_pp_sr":          bat_sr * 0.93,
                    "bat_death_sr":       bat_sr * 1.07,
                    "bat_boundary_pct":   40.0,
                    "bowl_economy":       bowl_eco,
                    "bowl_wkts_per_over": 0.24,
                    "bowl_pp_economy":    bowl_eco * 1.05,
                    "bowl_death_economy": bowl_eco * 1.12,
                    "bowl_dot_pct":       37.0,
                    "role_code":          float(rc),
                    "is_overseas":        float(proxy.get("is_overseas", 0)),
                    "is_pace":            float(any(w in (proxy.get("bowling_style","") or "").lower()
                                                  for w in ("fast","seam","swing"))),
                    "is_spin":            float(any(w in (proxy.get("bowling_style","") or "").lower()
                                                  for w in ("spin","off","leg","break","orthodox"))),
                    "venue_pace_economy": venue_info.get("venue_pace_economy", 8.5),
                    "venue_spin_economy": venue_info.get("venue_spin_economy", 8.0),
                    "innings_num":        float(innings_num),
                }
                X = np.array([[pf_t[c] for c in FEATURE_COLS]], dtype=np.float32)
                base = float(transfer.predict(X).flatten()[0])
                is_spin_p = bool(pf_t["is_spin"])
                is_pace_p = bool(pf_t["is_pace"])
                if rc in (2, 3):
                    if is_spin_p:
                        base *= spinner_penalty
                    if is_pace_p:
                        base *= (swing_bonus * 0.5 + pace_bounce * 0.5)
                if _source_out is not None:
                    _source_out.append("transfer")
                return round(float(np.clip(base, 0.0, 100.0)), 2)
            except Exception:
                pass  # fall through to analytical

        # Use T20 career stats as a proxy rather than a flat generic score.
        # Confidence is reduced vs PSL stats: tier 3 (no PSL) = 0.55, tier 2 (limited) = 0.70
        # Raised tier confidence to reduce regression-to-mean for overseas players.
        # Old: {1:0.85, 2:0.70, 3:0.55} — collapsed world-class overseas into 38-42 range.
        # New: tier1=PSL data (92% trust), tier2=IPL/limited (82%), tier3=generic T20 (65%).
        confidence = {1: 0.92, 2: 0.82, 3: 0.65}.get(tier, 0.65)

        bat_sr  = float(proxy.get("bat_sr",       120.0) or 120.0)
        bat_avg = float(proxy.get("bat_avg",        15.0) or 15.0)
        bowl_eco= float(proxy.get("bowl_economy",    8.5) or 8.5)

        # Build a minimal feature dict from career stats
        pf_proxy = {
            "bat_avg":            bat_avg,
            "bat_sr":             bat_sr,
            "bat_pp_sr":          bat_sr * 0.93,    # PP SR slightly below career
            "bat_death_sr":       bat_sr * 1.07,    # death SR slightly above career
            "bat_boundary_pct":   40.0,             # generic
            "bowl_economy":       bowl_eco,
            "bowl_wkts_per_over": 0.24,             # league average proxy
            "bowl_pp_economy":    bowl_eco * 1.05,  # PP slightly harder
            "bowl_death_economy": bowl_eco * 1.12,  # death slightly harder
            "bowl_dot_pct":       37.0,             # league average proxy
        }

        if rc == 3:
            raw = _bowling_score(pf_proxy)
        elif rc == 2:
            raw = (_batting_score(pf_proxy) + _bowling_score(pf_proxy)) / 2.0
        else:
            raw = _batting_score(pf_proxy)

        # Role-specific generic floor: bowlers anchor higher (reliable avg bowling is easier
        # to establish), pure batters anchor lower to give good stats more room to shine.
        # Old GENERIC_AVG=42.0 for all roles collapsed Steve Smith (raw 47) to 38.4.
        GENERIC_FLOOR = {0: 34.0, 1: 36.0, 2: 38.0, 3: 40.0}  # Batsman/WK/AR/Bowler
        generic_avg = GENERIC_FLOOR.get(rc, 36.0)
        base = raw * confidence + generic_avg * (1.0 - confidence)

        # Apply weather modifiers — ONLY for bowling roles (rc 2=AR, 3=Bowler).
        # A batter who happens to bowl spin (e.g. Steve Smith, leg-break) should NOT
        # have their batting score penalised by dew conditions.
        if rc in (2, 3):
            is_pace_p = any(w in (proxy.get("bowling_style","") or "").lower()
                            for w in ("fast","seam","swing"))
            is_spin_p = any(w in (proxy.get("bowling_style","") or "").lower()
                            for w in ("spin","off","leg","break","orthodox","chinaman","googly"))
            if is_spin_p:
                base *= spinner_penalty
            if is_pace_p:
                base *= (swing_bonus * 0.5 + pace_bounce * 0.5)

        if _source_out is not None:
            _source_out.append("analytical")
        return round(float(np.clip(base, 0.0, 100.0)), 2)

    pf = lookup[player]

    # --- TabNet path (UPGRADE 2) ---
    tabnet = payload.get("tabnet_model")
    if tabnet is not None:
        try:
            venue_info = payload.get("venue_lookup", {}).get(venue, {})
            row = {
                "bat_avg":            pf.get("bat_avg",           15.0),
                "bat_sr":             pf.get("bat_sr",           120.0),
                "bat_pp_sr":          pf.get("bat_pp_sr",        120.0),
                "bat_death_sr":       pf.get("bat_death_sr",     120.0),
                "bat_boundary_pct":   pf.get("bat_boundary_pct",  40.0),
                "bowl_economy":       pf.get("bowl_economy",       9.0),
                "bowl_wkts_per_over": pf.get("bowl_wkts_per_over", 0.0),
                "bowl_pp_economy":    pf.get("bowl_pp_economy",    8.5),
                "bowl_death_economy": pf.get("bowl_death_economy", 9.5),
                "bowl_dot_pct":       pf.get("bowl_dot_pct",      35.0),
                "role_code":          float(rc),
                "is_overseas":        float(pf.get("is_overseas", 0)),
                "is_pace":            float(pf.get("is_pace",     0)),
                "is_spin":            float(pf.get("is_spin",     0)),
                "venue_pace_economy": venue_info.get("venue_pace_economy", 8.5),
                "venue_spin_economy": venue_info.get("venue_spin_economy", 8.0),
                "innings_num":        float(innings_num),
            }
            X = np.array([[row[c] for c in FEATURE_COLS]], dtype=np.float32)
            base = float(tabnet.predict(X).flatten()[0])
            # Apply spinner_penalty post-prediction (same as current behaviour)
            is_spin = bool(pf.get("is_spin", 0))
            is_pace = bool(pf.get("is_pace", 0))
            if rc in (2, 3):
                if is_spin:
                    base *= spinner_penalty
                if is_pace:
                    base *= (swing_bonus * 0.5 + pace_bounce * 0.5)
            if _source_out is not None:
                _source_out.append("tabnet")
            return round(float(np.clip(base, 0.0, 100.0)), 2)
        except Exception:
            pass   # fall through to analytical

    # --- Analytical / XGBoost fallback ---
    if rc == 3:            # Pure bowler — bowling score only
        base = _bowling_score(pf)
    elif rc == 2:          # All-rounder — equal weight of both
        base = (_batting_score(pf) + _bowling_score(pf)) / 2.0
    else:                  # Batsman / WK-Batsman — batting score only
        base = _batting_score(pf)

    is_spin = bool(pf.get("is_spin", 0))
    is_pace = bool(pf.get("is_pace", 0))

    if rc in (2, 3):
        if is_spin:
            base *= spinner_penalty
        if is_pace:
            base *= (swing_bonus * 0.5 + pace_bounce * 0.5)

    if _source_out is not None:
        _source_out.append("standard")
    return round(float(np.clip(base, 0.0, 100.0)), 2)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    payload = train(verbose=True)
    meta    = payload["meta"]

    print(f"\n{'='*58}")
    print(f"  xi_scorer.pkl -- summary")
    print(f"{'='*58}")
    print(f"  Players in lookup : {meta['n_players']}")
    print(f"  Train rows        : {meta['train_rows']:,}")
    print(f"  Test rows         : {meta['test_rows']:,}")
    print(f"  Test RMSE         : {meta['rmse_test']:.2f} / 100")
    print(f"  Test MAE          : {meta['mae_test']:.2f} / 100")
    print(f"  Train seasons     : {meta['train_seasons']}")
    print(f"  Test seasons      : {meta['test_seasons']}")

    payload2 = load_model()
    active   = "TabNet (xi_tabnet.zip)" if payload2.get("tabnet_model") else "XGBoost (xi_scorer.pkl)"
    print(f"\n  Active inference model: {active}")

    demo_players = [
        "Babar Azam", "Shaheen Shah Afridi", "Shadab Khan",
        "Mohammad Rizwan", "Fakhar Zaman", "Imad Wasim",
    ]
    print(f"\n  Sample scores (National Stadium, Karachi, innings 1, no weather mod):")
    print(f"  {'Player':<28}  Score  Role")
    print(f"  {'-'*50}")
    for p in demo_players:
        sc   = score_player(p, "National Stadium, Karachi", 1, payload2)
        role = payload2["player_meta"].get(p, {}).get("role", "?")
        print(f"  {p:<28}  {sc:>5.1f}  {role}")

    print(f"\n  Spinner dew penalty demo (spinner_penalty=0.6):")
    for p in ["Shadab Khan", "Imad Wasim"]:
        nodew = score_player(p, "National Stadium, Karachi", 1, payload2)
        dew   = score_player(p, "National Stadium, Karachi", 1, payload2, spinner_penalty=0.6)
        print(f"  {p:<28}  no dew: {nodew:.1f}  dew: {dew:.1f}  drop: {nodew-dew:.1f}")

    print(f"{'='*58}\n")
