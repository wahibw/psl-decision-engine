# models/train_player_embeddings.py
# Upgrade 6: Trains 32-dim player embeddings from PSL partnership history.
#
# Purpose:
#   When two batters have no direct PSL partnership history, the embedding model
#   finds the most similar KNOWN pair (cosine similarity on combined embeddings)
#   and uses their historical stats as a proxy — better than generic league averages.
#
# Model:  Simple encoder/decoder (6-dim profile → 32-dim embedding → 6-dim reconstruction)
# Input:  Per-player aggregated profile from partnership_history.parquet
# Output: models/saved/player_embeddings.pt  (dict: player_name -> 32-dim np.ndarray)
#         models/saved/pair_embeddings.pt     (dict: (b1,b2) -> 32-dim combined vector)
#
# Run:  python models/train_player_embeddings.py

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT   = Path(__file__).resolve().parent.parent
MODELS_DIR  = PROJ_ROOT / "models" / "saved"
DATA_DIR    = PROJ_ROOT / "data" / "processed"
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

HISTORY_PATH      = DATA_DIR / "partnership_history.parquet"
EMBED_PATH        = MODELS_DIR / "player_embeddings.pt"
PAIR_EMBED_PATH   = MODELS_DIR / "pair_embeddings.pt"

EMBED_DIM         = 32
MAX_EMBED_EPOCHS  = 10   # fast CPU training


# ---------------------------------------------------------------------------
# PROFILE BUILDER
# ---------------------------------------------------------------------------

def _build_player_profiles(history: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate partnership_history per player into a 6-feature profile vector.

    Features (all normalised to approx [0,1] range before embedding training):
      avg_runs       — mean partnership runs this player contributes
      avg_sr         — mean partnership SR
      pace_pct       — % of their partnerships broken by pace
      spin_pct       — % broken by spin
      change_pct     — % broken by bowling change
      avg_balls      — mean partnership length in balls
    """
    career = history[history["season"] == 0].copy()

    # Stack both batter columns so each row appears for each player
    rows = []
    for _, r in career.iterrows():
        for batter in (r["batter1"], r["batter2"]):
            rows.append({
                "player":      batter,
                "avg_runs":    float(r.get("avg_runs",  0) or 0),
                "avg_balls":   float(r.get("avg_balls", 0) or 0),
                "pace_pct":    float(r.get("broken_by_pace_pct",           0) or 0),
                "spin_pct":    float(r.get("broken_by_spin_pct",           0) or 0),
                "change_pct":  float(r.get("broken_by_bowling_change_pct", 0) or 0),
                "occurrences": float(r.get("occurrences", 1) or 1),
            })

    df = pd.DataFrame(rows)
    df["avg_sr"] = np.where(
        df["avg_balls"] > 0,
        df["avg_runs"] / df["avg_balls"] * 100,
        120.0,
    )

    # Weighted average per player (weight by occurrences)
    profiles = (
        df.groupby("player")
        .apply(lambda g: pd.Series({
            "avg_runs":   np.average(g["avg_runs"],  weights=g["occurrences"]),
            "avg_sr":     np.average(g["avg_sr"],    weights=g["occurrences"]),
            "avg_balls":  np.average(g["avg_balls"], weights=g["occurrences"]),
            "pace_pct":   np.average(g["pace_pct"],  weights=g["occurrences"]),
            "spin_pct":   np.average(g["spin_pct"],  weights=g["occurrences"]),
            "change_pct": np.average(g["change_pct"],weights=g["occurrences"]),
        }), include_groups=False)
        .reset_index()
    )
    return profiles


# ---------------------------------------------------------------------------
# NORMALISE
# ---------------------------------------------------------------------------

def _normalise(profiles: pd.DataFrame) -> tuple[np.ndarray, dict, dict]:
    """Min-max normalise feature columns. Returns (X, means, scales)."""
    FEAT_COLS = ["avg_runs", "avg_sr", "avg_balls", "pace_pct", "spin_pct", "change_pct"]
    X = profiles[FEAT_COLS].values.astype(np.float32)
    means  = X.mean(axis=0)
    scales = X.std(axis=0) + 1e-8
    X = (X - means) / scales
    return X, dict(zip(FEAT_COLS, means)), dict(zip(FEAT_COLS, scales))


# ---------------------------------------------------------------------------
# EMBEDDING MODEL (PyTorch autoencoder)
# ---------------------------------------------------------------------------

def _build_model(input_dim: int = 6, embed_dim: int = EMBED_DIM):
    try:
        import torch.nn as nn
    except ImportError:
        return None

    class _EmbedNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, embed_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z

    return _EmbedNet()


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------

def train(verbose: bool = True) -> bool:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:
        print(f"[train_player_embeddings] Missing dependency: {e}")
        return False

    if not HISTORY_PATH.exists():
        print(f"[train_player_embeddings] partnership_history.parquet not found")
        return False

    history  = pd.read_parquet(HISTORY_PATH)
    profiles = _build_player_profiles(history)

    if len(profiles) < 5:
        print(f"[train_player_embeddings] Not enough players for embedding ({len(profiles)})")
        return False

    if verbose:
        print(f"[train_player_embeddings] {len(profiles)} player profiles built")

    X, means, scales = _normalise(profiles)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset  = TensorDataset(X_tensor)
    loader   = DataLoader(dataset, batch_size=32, shuffle=True)

    model = _build_model(input_dim=X.shape[1], embed_dim=EMBED_DIM)
    if model is None:
        return False

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, MAX_EMBED_EPOCHS + 1):
        total_loss = 0.0
        for (batch,) in loader:
            optimiser.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == MAX_EMBED_EPOCHS):
            print(f"[train_player_embeddings] Epoch {epoch}/{MAX_EMBED_EPOCHS}  loss={total_loss/len(loader):.4f}")

    # Extract final embeddings for every player
    model.eval()
    with torch.no_grad():
        _, embeddings = model(X_tensor)
        emb_np = embeddings.numpy()

    player_names = profiles["player"].tolist()
    player_emb   = {name: emb_np[i] for i, name in enumerate(player_names)}

    # Build pair embeddings for all known career pairs (batter1+batter2 combined)
    career = history[history["season"] == 0]
    pair_emb: dict = {}
    for _, row in career.iterrows():
        b1, b2 = row["batter1"], row["batter2"]
        if b1 in player_emb and b2 in player_emb:
            combined = player_emb[b1] + player_emb[b2]
            norm = np.linalg.norm(combined)
            pair_emb[(b1, b2)] = combined / (norm + 1e-8)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(player_emb, str(EMBED_PATH))
    torch.save(pair_emb,   str(PAIR_EMBED_PATH))

    if verbose:
        print(f"[train_player_embeddings] {len(player_emb)} player embeddings saved -> {EMBED_PATH}")
        print(f"[train_player_embeddings] {len(pair_emb)} pair embeddings saved  -> {PAIR_EMBED_PATH}")

    # Quick similarity check
    if "Babar Azam" in player_emb and "Mohammad Rizwan" in player_emb:
        e1 = player_emb["Babar Azam"]
        e2 = player_emb["Mohammad Rizwan"]
        cos = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))
        print(f"[train_player_embeddings] Similarity check: Babar/Rizwan cosine={cos:.3f}")

    return True


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ok = train(verbose=True)
    status = "Done" if ok else "Failed"
    print(f"\n[train_player_embeddings] {status} — player_embeddings.pt ready for partnership_engine.py")
