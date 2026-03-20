# models/train_bowling.py
# Trains a phase-aware XGBoost regressor that predicts runs conceded per over.
# UPGRADE 1: Also trains a 2-layer LSTM (bowling_lstm.pt) as primary model;
#            XGBoost bowling_model.pkl becomes the fallback.
# Used by bowling_plan.py to rank available bowlers per phase.
#
# Training row: one over bowled by one bowler in a PSL match
# Features: phase, over_num, innings, match context + bowler career phase stats
# Target: runs_in_over (capped at 28 to reduce extreme outlier influence)
#
# Output: models/saved/bowling_model.pkl  (XGBRegressor fallback)
#         models/saved/bowling_lstm.pt    (LSTM primary, preferred at runtime)
#
# Run:  python models/train_bowling.py

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
OUTPUT_FILE   = MODELS_DIR / "bowling_model.pkl"
LSTM_FILE     = MODELS_DIR / "bowling_lstm.pt"

BBB_PATH     = PROCESSED_DIR / "ball_by_ball.parquet"
STATS_PATH   = PROCESSED_DIR / "player_stats.parquet"
PLAYER_INDEX = PROCESSED_DIR / "player_index_2026_enriched.csv"
PLAYER_INDEX_FALLBACK = PROJ_ROOT.parent / "player_index_2026_enriched.csv"

# Target cap (runs per over): extreme outliers distort the regressor
RUNS_CAP = 28

# Minimum overs bowled (career) before a bowler gets their own feature row
MIN_OVERS_FOR_STATS = 10

# Phase code mapping
PHASE_CODE = {"powerplay": 0, "middle": 1, "death": 2, "super_over": 2}

# Feature names (must match exactly in production scoring)
FEATURE_COLS = [
    "phase_code",
    "over_num",
    "innings",
    "score_rate",          # crr at start of this over
    "wickets_lost",        # cumulative wickets before this over
    "bowler_pp_econ",      # bowler career economy in powerplay
    "bowler_mid_econ",     # bowler career economy in middle
    "bowler_death_econ",   # bowler career economy in death
    "bowler_pp_wkts_po",   # wickets per over in powerplay
    "bowler_mid_wkts_po",  # wickets per over in middle
    "bowler_death_wkts_po",
    "bowler_is_pace",
    "bowler_is_spin",
    "bowler_pp_overs",     # career overs in phase (data volume proxy)
    "bowler_mid_overs",
    "bowler_death_overs",
]


# ---------------------------------------------------------------------------
# BOWL TYPE LOOKUP
# ---------------------------------------------------------------------------

def _load_bowl_types(pi_path: Path) -> dict[str, str]:
    """Returns {player_name: 'pace'|'spin'|'unknown'}."""
    bowl_types: dict[str, str] = {}
    if not pi_path.exists():
        return bowl_types
    with open(pi_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name  = row.get("player_name", "").strip()
            style = (row.get("bowling_style") or "").lower().strip()
            if not name:
                continue
            if any(w in style for w in ("fast", "medium", "pace", "seam", "swing")):
                bowl_types[name] = "pace"
            elif any(w in style for w in ("spin", "off", "leg", "googly", "chinaman", "slow")):
                bowl_types[name] = "spin"
            else:
                bowl_types[name] = "unknown"
    return bowl_types


# ---------------------------------------------------------------------------
# BUILD OVER-LEVEL TRAINING DATA
# ---------------------------------------------------------------------------

def _build_over_level(bbb: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ball_by_ball to one row per (match_id, innings, over, bowler).
    Adds running score/wickets at the START of that over.
    """
    # We'll compute running totals by match+innings, then take per-over view
    bbb = bbb.copy()
    bbb["phase_code"] = bbb["phase"].map(PHASE_CODE).fillna(1).astype(int)

    # Sort for correct running totals
    bbb = bbb.sort_values(["match_id", "innings", "over", "ball"]).reset_index(drop=True)

    # Per over: aggregate
    over_grp = (
        bbb.groupby(["match_id", "season", "innings", "over", "bowler"], observed=True)
        .agg(
            over_runs        = ("runs_total",  "sum"),
            over_wickets     = ("is_wicket",   "sum"),
            phase_code       = ("phase_code",  "first"),
            # Score at end of over (we'll derive start-of-over below)
            innings_score_end  = ("innings_score",   "last"),
            innings_wickets_end= ("innings_wickets", "last"),
            legal_balls      = ("is_wide",     lambda x: (~x).sum()),
        )
        .reset_index()
    )
    over_grp["over_runs"] = over_grp["over_runs"].clip(upper=RUNS_CAP).astype(float)

    # Derive score/wickets at START of over:
    # score_end - over_runs = score at start
    over_grp["score_start"]   = (over_grp["innings_score_end"] - over_grp["over_runs"]).clip(lower=0)
    over_grp["wickets_start"] = (over_grp["innings_wickets_end"] - over_grp["over_wickets"]).clip(lower=0)

    # CRR at start of over (score_start / overs_done, where overs_done = over_num)
    over_grp["score_rate"] = np.where(
        over_grp["over"] > 0,
        (over_grp["score_start"] / over_grp["over"]).clip(upper=20.0),
        0.0,
    )

    return over_grp


# ---------------------------------------------------------------------------
# BUILD BOWLER FEATURE LOOKUP
# ---------------------------------------------------------------------------

def _build_bowler_features(
    stats: pd.DataFrame,
    bowl_types: dict[str, str],
) -> pd.DataFrame:
    """
    Returns a DataFrame keyed by bowler (career stats only, season=0).
    Contains economy and wickets_per_over per phase.
    """
    career = stats[stats["season"] == 0].copy()

    def _get(player: str, phase: str, col: str, default: float) -> float:
        row = career[(career["player_name"] == player) & (career["phase"] == phase)]
        if row.empty:
            return default
        v = row.iloc[0].get(col)
        return float(v) if pd.notna(v) else default

    # Collect all unique bowlers
    bowlers = career["player_name"].unique()

    records = []
    for bowler in bowlers:
        # Overs bowled per phase (proxy for data reliability)
        pp_balls  = career[(career["player_name"] == bowler) & (career["phase"] == "powerplay")]["bowl_balls"].sum()
        mid_balls = career[(career["player_name"] == bowler) & (career["phase"] == "middle")]["bowl_balls"].sum()
        dth_balls = career[(career["player_name"] == bowler) & (career["phase"] == "death")]["bowl_balls"].sum()

        pp_overs  = pp_balls  / 6
        mid_overs = mid_balls / 6
        dth_overs = dth_balls / 6

        total_overs = pp_overs + mid_overs + dth_overs
        if total_overs < MIN_OVERS_FOR_STATS:
            continue

        btype = bowl_types.get(bowler, "unknown")

        def _wkts_po(phase: str) -> float:
            row = career[(career["player_name"] == bowler) & (career["phase"] == phase)]
            if row.empty:
                return 0.3
            overs = float(row.iloc[0].get("bowl_overs") or 0)
            wkts  = float(row.iloc[0].get("bowl_wickets") or 0)
            return round(wkts / overs, 3) if overs > 0 else 0.3

        records.append({
            "bowler":              bowler,
            "bowler_pp_econ":      _get(bowler, "powerplay", "bowl_economy",    8.0),
            "bowler_mid_econ":     _get(bowler, "middle",    "bowl_economy",    7.5),
            "bowler_death_econ":   _get(bowler, "death",     "bowl_economy",    9.5),
            "bowler_pp_wkts_po":   _wkts_po("powerplay"),
            "bowler_mid_wkts_po":  _wkts_po("middle"),
            "bowler_death_wkts_po":_wkts_po("death"),
            "bowler_is_pace":      int(btype == "pace"),
            "bowler_is_spin":      int(btype == "spin"),
            "bowler_pp_overs":     round(pp_overs,  1),
            "bowler_mid_overs":    round(mid_overs, 1),
            "bowler_death_overs":  round(dth_overs, 1),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# BUILD FULL FEATURE MATRIX
# ---------------------------------------------------------------------------

def _build_features(
    over_df:     pd.DataFrame,
    bowler_feat: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Join over-level data with bowler career features.
    Returns (X, y).
    """
    df = over_df.merge(bowler_feat, on="bowler", how="inner")

    df["over_num"] = df["over"].astype(int)
    df["innings"]  = df["innings"].astype(int)
    df["wickets_lost"] = df["wickets_start"].astype(float)

    X = df[FEATURE_COLS].astype(float)
    y = df["over_runs"].astype(float)

    return X, y


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------

def train(
    bbb_path:    Optional[Path] = None,
    stats_path:  Optional[Path] = None,
    output_path: Optional[Path] = None,
    verbose:     bool = True,
) -> dict:
    """
    Train the bowling model and save to disk.

    Returns:
        dict with keys: model, feature_cols, meta
    """
    t0 = time.time()

    bbb_path    = Path(bbb_path)    if bbb_path    else BBB_PATH
    stats_path  = Path(stats_path)  if stats_path  else STATS_PATH
    output_path = Path(output_path) if output_path else OUTPUT_FILE

    if verbose:
        print("[train_bowling] Loading data...")

    bbb   = pd.read_parquet(bbb_path)
    stats = pd.read_parquet(stats_path)

    pi_path = PLAYER_INDEX if PLAYER_INDEX.exists() else PLAYER_INDEX_FALLBACK
    bowl_types = _load_bowl_types(pi_path) if pi_path.exists() else {}

    if verbose:
        print(f"[train_bowling] {len(bbb):,} deliveries loaded")

    # Build training data
    if verbose:
        print("[train_bowling] Building over-level dataset...")
    over_df = _build_over_level(bbb)

    if verbose:
        print("[train_bowling] Building bowler feature lookup...")
    bowler_feat = _build_bowler_features(stats, bowl_types)

    if verbose:
        print(f"[train_bowling] {len(bowler_feat)} bowlers with sufficient data")

    # Build merged dataset for train/test split
    over_merged = over_df.merge(bowler_feat, on="bowler", how="inner").copy()
    over_merged["over_num"]    = over_merged["over"].astype(int)
    over_merged["innings"]     = over_merged["innings"].astype(int)
    over_merged["wickets_lost"]= over_merged["wickets_start"].astype(float)

    X = over_merged[FEATURE_COLS].astype(float)
    y = over_merged["over_runs"].astype(float)

    if verbose:
        print(f"[train_bowling] Training set: {len(X):,} rows x {len(FEATURE_COLS)} features")
        print(f"[train_bowling]   Target: mean={y.mean():.2f}, std={y.std():.2f}, "
              f"min={y.min():.0f}, max={y.max():.0f}")

    # Train/test split by season (hold out last 2 seasons)
    all_seasons = sorted(over_merged["season"].unique())
    test_seasons = all_seasons[-2:] if len(all_seasons) >= 2 else all_seasons[-1:]
    train_seasons = [s for s in all_seasons if s not in test_seasons]

    train_mask = over_merged["season"].isin(train_seasons)
    test_mask  = over_merged["season"].isin(test_seasons)

    X_train = over_merged.loc[train_mask, FEATURE_COLS].astype(float)
    y_train = over_merged.loc[train_mask, "over_runs"].astype(float)
    X_test  = over_merged.loc[test_mask,  FEATURE_COLS].astype(float)
    y_test  = over_merged.loc[test_mask,  "over_runs"].astype(float)

    if verbose:
        print(f"[train_bowling] Train: {len(X_train):,} overs  |  Test: {len(X_test):,} overs")
        print(f"[train_bowling]   Train seasons: {train_seasons}")
        print(f"[train_bowling]   Test seasons:  {test_seasons}")

    # XGBoost regressor
    model = XGBRegressor(
        n_estimators    = 300,
        max_depth       = 4,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        min_child_weight= 5,
        reg_alpha       = 0.1,
        reg_lambda      = 1.0,
        random_state    = 42,
        verbosity       = 0,
        n_jobs          = -1,
    )

    if verbose:
        print("[train_bowling] Fitting XGBRegressor...")

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Evaluate
    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)

    rmse_train = float(np.sqrt(np.mean((pred_train - y_train) ** 2)))
    rmse_test  = float(np.sqrt(np.mean((pred_test  - y_test)  ** 2)))
    mae_test   = float(np.mean(np.abs(pred_test - y_test)))

    if verbose:
        print(f"[train_bowling] Train RMSE: {rmse_train:.3f}  |  Test RMSE: {rmse_test:.3f}  |  MAE: {mae_test:.3f}")

    # Feature importance
    imp = dict(zip(FEATURE_COLS, model.feature_importances_))
    top5 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:5]
    if verbose:
        print("[train_bowling] Top 5 features:")
        for fname, fimpt in top5:
            print(f"    {fname:<30}  {fimpt:.4f}")

    # Build bowler scoring lookup for production use
    # Maps bowler_name -> feature dict (for scoring without needing full stats load)
    bowler_lookup = bowler_feat.set_index("bowler").to_dict("index")

    # Save
    payload = {
        "model":          model,
        "feature_cols":   FEATURE_COLS,
        "bowler_lookup":  bowler_lookup,
        "bowl_types":     bowl_types,
        "phase_code":     PHASE_CODE,
        "meta": {
            "rmse_train":   rmse_train,
            "rmse_test":    rmse_test,
            "mae_test":     mae_test,
            "train_rows":   len(X_train),
            "test_rows":    len(X_test),
            "train_seasons":train_seasons,
            "test_seasons": test_seasons,
            "n_bowlers":    len(bowler_lookup),
            "runs_cap":     RUNS_CAP,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    elapsed = time.time() - t0
    size_kb = output_path.stat().st_size / 1024
    if verbose:
        print(f"[train_bowling] Saved -> {output_path}  ({size_kb:.0f} KB)  [{elapsed:.1f}s]")

    # --- UPGRADE 1: Also train the LSTM ---
    train_lstm(over_df, bowler_feat, verbose=verbose)

    return payload


# ---------------------------------------------------------------------------
# UPGRADE 1 — LSTM MODEL
# ---------------------------------------------------------------------------
# Sequence features per time step (6 dims):
#   [runs_this_over, wickets_this_over, bowler_type_enc, phase_enc,
#    cumulative_score, crr]
LSTM_INPUT_SIZE  = 6
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS  = 2
LSTM_DROPOUT     = 0.3
LSTM_SEQ_LEN     = 5          # look-back window (overs)
LSTM_EARLY_STOP  = 10         # patience for early stopping

_BOWLER_TYPE_ENC = {"pace": 1.0, "spin": 2.0, "unknown": 0.0}


def _try_import_torch():
    """Return torch module or None if not installed."""
    try:
        import torch
        return torch
    except ImportError:
        return None


class BowlingLSTM:
    """
    Thin wrapper around a PyTorch LSTM so it can be used without importing
    torch at module load time (torch is optional — falls back to XGBoost).
    """
    def __init__(self, state_dict, scaler_mean, scaler_std,
                 input_size=LSTM_INPUT_SIZE, hidden_size=LSTM_HIDDEN_SIZE,
                 num_layers=LSTM_NUM_LAYERS, dropout=LSTM_DROPOUT):
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                    dropout=dropout, batch_first=True)
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        self._net = _Net()
        self._net.load_state_dict(state_dict)
        self._net.eval()
        self._mean = torch.tensor(scaler_mean, dtype=torch.float32)
        self._std  = torch.tensor(scaler_std,  dtype=torch.float32)
        self._torch = torch

    def predict(self, x_np: np.ndarray) -> float:
        """x_np: (seq_len, input_size) float32 array → scalar prediction."""
        t = self._torch.tensor(x_np, dtype=self._torch.float32).unsqueeze(0)
        t = (t - self._mean) / (self._std + 1e-8)
        with self._torch.no_grad():
            return float(self._net(t).item())


def _build_lstm_sequences(
    over_df: pd.DataFrame,
    bowler_feat: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences of length LSTM_SEQ_LEN from over-level data.
    Returns (X, y) where X.shape = (N, LSTM_SEQ_LEN, LSTM_INPUT_SIZE).
    """
    df = over_df.merge(bowler_feat, on="bowler", how="inner").copy()
    df["over_num"]   = df["over"].astype(int)
    df["innings"]    = df["innings"].astype(int)
    df["wickets_lst"]= df["wickets_start"].astype(float)

    df["bowler_type_enc"] = df["bowler"].map(
        lambda b: _BOWLER_TYPE_ENC.get(
            "pace" if df.loc[df["bowler"] == b, "bowler_is_pace"].iloc[0] == 1
            else ("spin" if df.loc[df["bowler"] == b, "bowler_is_spin"].iloc[0] == 1
                  else "unknown"), 0.0)
    ).fillna(0.0)
    df["phase_enc"] = df["phase_code"].astype(float)

    X_list, y_list = [], []
    for (match_id, innings), grp in df.groupby(["match_id", "innings"], observed=True):
        grp = grp.sort_values("over_num").reset_index(drop=True)
        if len(grp) < LSTM_SEQ_LEN + 1:
            continue
        for i in range(LSTM_SEQ_LEN, len(grp)):
            window = grp.iloc[i - LSTM_SEQ_LEN: i]
            seq = window[["over_runs", "over_wickets",
                          "bowler_type_enc", "phase_enc",
                          "score_start", "score_rate"]].values.astype(np.float32)
            # Use bowler-type enc of the TARGET over (next bowler we're predicting for)
            target_row = grp.iloc[i]
            target_seq = seq.copy()
            # Replace last step's bowler type with current bowler
            target_seq[-1, 2] = float(_BOWLER_TYPE_ENC.get(
                "pace" if target_row.get("bowler_is_pace", 0) == 1 else
                ("spin" if target_row.get("bowler_is_spin", 0) == 1 else "unknown"), 0.0
            ))
            X_list.append(target_seq)
            y_list.append(float(grp.iloc[i]["over_runs"]))

    if not X_list:
        return np.empty((0, LSTM_SEQ_LEN, LSTM_INPUT_SIZE)), np.empty(0)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def train_lstm(
    over_df:     pd.DataFrame,
    bowler_feat: pd.DataFrame,
    output_path: Optional[Path] = None,
    verbose:     bool = True,
) -> Optional[dict]:
    """
    Train the LSTM model and save to bowling_lstm.pt.
    Returns the LSTM payload dict, or None if torch is unavailable.
    """
    torch = _try_import_torch()
    if torch is None:
        if verbose:
            print("[train_bowling] torch not available — skipping LSTM training")
        return None

    import torch.nn as nn

    output_path = Path(output_path) if output_path else LSTM_FILE

    if verbose:
        print("[train_bowling] Building LSTM sequences...")

    X, y = _build_lstm_sequences(over_df, bowler_feat)
    if len(X) < 50:
        if verbose:
            print("[train_bowling] Too few sequences for LSTM — skipping")
        return None

    # Normalisation (fit on train split)
    n_train = int(len(X) * 0.85)
    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]

    mean = X_tr.mean(axis=(0, 1), keepdims=True)   # (1, 1, 6)
    std  = X_tr.std(axis=(0, 1), keepdims=True) + 1e-8

    X_tr_n = (X_tr - mean) / std
    X_te_n = (X_te - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[train_bowling] LSTM training on {device}  sequences={len(X)}")

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(LSTM_INPUT_SIZE, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS,
                                dropout=LSTM_DROPOUT, batch_first=True)
            self.head = nn.Linear(LSTM_HIDDEN_SIZE, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    net   = _Net().to(device)
    opt   = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    Xtr_t = torch.tensor(X_tr_n, dtype=torch.float32).to(device)
    ytr_t = torch.tensor(y_tr,   dtype=torch.float32).to(device)
    Xte_t = torch.tensor(X_te_n, dtype=torch.float32).to(device)
    yte_t = torch.tensor(y_te,   dtype=torch.float32).to(device)

    best_val, patience_count, best_sd = float("inf"), 0, None
    batch = 256
    epochs = 30   # reduced from 200 for fast CPU training (early stopping still active)

    for epoch in range(1, epochs + 1):
        net.train()
        idx = torch.randperm(len(Xtr_t))
        for s in range(0, len(Xtr_t), batch):
            b_idx = idx[s: s + batch]
            opt.zero_grad()
            loss = loss_fn(net(Xtr_t[b_idx]), ytr_t[b_idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(net(Xte_t), yte_t).item())
        if val_loss < best_val:
            best_val = val_loss
            best_sd  = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= LSTM_EARLY_STOP:
                if verbose:
                    print(f"[train_bowling] LSTM early stop at epoch {epoch}  val_loss={best_val:.4f}")
                break
        if verbose and epoch % 20 == 0:
            print(f"[train_bowling] LSTM epoch {epoch:3d}  val_loss={val_loss:.4f}")

    # RMSE metrics
    net.load_state_dict(best_sd)
    net.eval()
    with torch.no_grad():
        pred_te = net(Xte_t).cpu().numpy()
    rmse = float(np.sqrt(np.mean((pred_te - y_te) ** 2)))
    mae  = float(np.mean(np.abs(pred_te - y_te)))

    if verbose:
        print(f"[train_bowling] LSTM Test RMSE={rmse:.3f}  MAE={mae:.3f}")

    payload = {
        "model_type":   "lstm",
        "state_dict":   best_sd,
        "scaler_mean":  mean.squeeze().tolist(),
        "scaler_std":   std.squeeze().tolist(),
        "meta": {"rmse_test": rmse, "mae_test": mae, "n_sequences": len(X)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    if verbose:
        print(f"[train_bowling] LSTM saved -> {output_path}")
    return payload


# ---------------------------------------------------------------------------
# SCORING HELPER  (imported by bowling_plan.py at runtime)
# ---------------------------------------------------------------------------

def load_model(model_path: Optional[Path] = None) -> dict:
    """
    Load the bowling model payload.
    Prefers bowling_lstm.pt (LSTM); falls back to bowling_model.pkl (XGBoost).
    Adds 'lstm_model' key to payload when LSTM is available.
    """
    # Try LSTM first
    lstm_path = LSTM_FILE
    if lstm_path.exists():
        torch = _try_import_torch()
        if torch is not None:
            try:
                lstm_payload = torch.load(lstm_path, map_location="cpu",
                                          weights_only=False)
                lstm_obj = BowlingLSTM(
                    state_dict  = lstm_payload["state_dict"],
                    scaler_mean = lstm_payload["scaler_mean"],
                    scaler_std  = lstm_payload["scaler_std"],
                )
                # Still load XGBoost payload for the bowler_lookup table
                pkl_path = Path(model_path) if model_path else OUTPUT_FILE
                if pkl_path.exists():
                    with open(pkl_path, "rb") as f:
                        base_payload = pickle.load(f)
                else:
                    base_payload = {
                        "bowler_lookup": {}, "bowl_types": {},
                        "phase_code": PHASE_CODE, "feature_cols": FEATURE_COLS,
                        "model": None, "meta": {},
                    }
                base_payload["lstm_model"]  = lstm_obj
                base_payload["lstm_meta"]   = lstm_payload.get("meta", {})
                return base_payload
            except Exception as e:
                print(f"[train_bowling] LSTM load failed ({e}) — using XGBoost fallback")

    # Fallback: XGBoost pkl
    model_path = Path(model_path) if model_path else OUTPUT_FILE
    with open(model_path, "rb") as f:
        return pickle.load(f)


def score_bowler(
    bowler:      str,
    phase:       str,
    over_num:    int,
    innings:     int,
    score_rate:  float,
    wickets_lost:int,
    payload:     dict,
) -> float:
    """
    Predict runs per over for one bowler in the given match context.
    Returns predicted_runs_per_over (lower = better).

    Uses LSTM (bowling_lstm.pt) if available, else XGBoost (bowling_model.pkl).
    Falls back to 9.0 if bowler not in lookup.
    """
    lookup         = payload["bowler_lookup"]
    phase_code_map = payload["phase_code"]

    if bowler not in lookup:
        return 9.0   # Unknown bowler — conservative default

    bf = lookup[bowler]

    # --- LSTM path ---
    lstm_model = payload.get("lstm_model")
    if lstm_model is not None:
        try:
            phase_enc   = float(phase_code_map.get(phase, 1))
            btype_enc   = float(_BOWLER_TYPE_ENC.get(
                "pace" if bf.get("bowler_is_pace", 0) else
                ("spin" if bf.get("bowler_is_spin", 0) else "unknown"), 0.0))
            cum_score   = score_rate * max(1, over_num - 1)
            # Build a LSTM_SEQ_LEN × LSTM_INPUT_SIZE context
            # We repeat the current context across all sequence steps
            # (LSTM sees a "flat" history when no ball-by-ball history is available)
            step = np.array([
                7.0,        # placeholder prior-over runs (league avg)
                0.3,        # placeholder prior-over wickets
                btype_enc,
                phase_enc,
                cum_score,
                score_rate,
            ], dtype=np.float32)
            seq = np.tile(step, (LSTM_SEQ_LEN, 1))
            # Last step uses actual context
            seq[-1] = np.array([
                wickets_lost * 2.0,   # crude proxy for pressure
                wickets_lost / max(1, over_num),
                btype_enc,
                phase_enc,
                cum_score,
                score_rate,
            ], dtype=np.float32)
            pred = lstm_model.predict(seq)
            return round(float(np.clip(pred, 0.0, RUNS_CAP)), 2)
        except Exception:
            pass   # fall through to XGBoost

    # --- XGBoost fallback ---
    feature_cols = payload["feature_cols"]
    model        = payload["model"]
    if model is None:
        return 9.0

    row = {
        "phase_code":          phase_code_map.get(phase, 1),
        "over_num":            over_num,
        "innings":             innings,
        "score_rate":          score_rate,
        "wickets_lost":        wickets_lost,
        "bowler_pp_econ":      bf.get("bowler_pp_econ",       8.0),
        "bowler_mid_econ":     bf.get("bowler_mid_econ",      7.5),
        "bowler_death_econ":   bf.get("bowler_death_econ",    9.5),
        "bowler_pp_wkts_po":   bf.get("bowler_pp_wkts_po",   0.3),
        "bowler_mid_wkts_po":  bf.get("bowler_mid_wkts_po",  0.3),
        "bowler_death_wkts_po":bf.get("bowler_death_wkts_po",0.3),
        "bowler_is_pace":      bf.get("bowler_is_pace",        0),
        "bowler_is_spin":      bf.get("bowler_is_spin",        0),
        "bowler_pp_overs":     bf.get("bowler_pp_overs",       0),
        "bowler_mid_overs":    bf.get("bowler_mid_overs",      0),
        "bowler_death_overs":  bf.get("bowler_death_overs",    0),
    }
    X = pd.DataFrame([row])[feature_cols].astype(float)
    return round(max(0.0, float(model.predict(X)[0])), 2)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    payload = train(verbose=True)
    meta    = payload["meta"]

    print(f"\n{'='*55}")
    print(f"  bowling_model.pkl -- summary")
    print(f"{'='*55}")
    print(f"  Bowlers in lookup : {meta['n_bowlers']}")
    print(f"  Train rows        : {meta['train_rows']:,}")
    print(f"  Test rows         : {meta['test_rows']:,}")
    print(f"  Test RMSE         : {meta['rmse_test']:.3f} runs/over")
    print(f"  Test MAE          : {meta['mae_test']:.3f} runs/over")
    print(f"  Train seasons     : {meta['train_seasons']}")
    print(f"  Test seasons      : {meta['test_seasons']}")

    # Reload so load_model() picks up the freshly saved LSTM
    payload2 = load_model()
    active   = "LSTM (bowling_lstm.pt)" if payload2.get("lstm_model") else "XGBoost (bowling_model.pkl)"
    print(f"\n  Active inference model: {active}")

    print(f"\n  Sample predictions (death overs, innings 2, score_rate=9, 3wkts):")
    demo_bowlers = ["Shaheen Shah Afridi", "Imad Wasim", "Mohammad Amir", "Shadab Khan"]
    for b in demo_bowlers:
        pred   = score_bowler(b, "death", 17, 2, 9.0, 3, payload2)
        status = "(in lookup)" if b in payload2["bowler_lookup"] else "(default)"
        print(f"    {b:<30}  predicted: {pred:.2f} runs/over  {status}")

    print(f"{'='*55}\n")
