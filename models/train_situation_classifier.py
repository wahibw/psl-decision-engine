# models/train_situation_classifier.py
# Fine-tunes distilbert-base-uncased to classify match situations:
#   CRITICAL (0) | WARNING (1) | INFO (2)
#
# Training data derived from ball_by_ball.parquet:
#   CRITICAL  = over where a wicket fell in the NEXT over (pressure moment)
#   WARNING   = over where 8+ runs scored in the NEXT over (big-over danger)
#   INFO      = all other overs
#
# Input format:
#   "Over {over}. Score {score}/{wickets}. Partnership {runs} off {balls}.
#    Phase {phase}. Dew active: {dew}. Bowler: {bowler_type}."
#
# Save: models/saved/situation_classifier/
# Run:  python models/train_situation_classifier.py

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

CLASSIFIER_PATH = MODELS_DIR / "situation_classifier"
LABEL_NAMES     = {0: "CRITICAL", 1: "WARNING", 2: "INFO"}
MIN_CONF        = 0.6   # minimum probability to use model vs rule-based fallback


# ---------------------------------------------------------------------------
# BUILD TRAINING DATA
# ---------------------------------------------------------------------------

def build_training_data(bbb_path: Path | None = None) -> pd.DataFrame:
    """
    Derive (text_input, label) rows from ball_by_ball.parquet.
    Returns a DataFrame with columns: text, label (0/1/2)
    """
    path = bbb_path or (DATA_DIR / "ball_by_ball.parquet")
    bbb  = pd.read_parquet(path)

    # Aggregate to over level: one row per (match_id, innings, over)
    over_grp = (
        bbb.groupby(["match_id", "innings", "over", "phase"], observed=True)
        .agg(
            runs_this_over    = ("runs_total",       "sum"),
            wickets_this_over = ("is_wicket",         "sum"),
            innings_score     = ("innings_score",    "last"),
            innings_wickets   = ("innings_wickets",  "last"),
        )
        .reset_index()
        .sort_values(["match_id", "innings", "over"])
    )

    # Derive NEXT-over events
    over_grp["wicket_next_over"] = (
        over_grp.groupby(["match_id", "innings"])["wickets_this_over"]
        .shift(-1)
        .fillna(0)
        .astype(int)
    )
    over_grp["runs_next_over"] = (
        over_grp.groupby(["match_id", "innings"])["runs_this_over"]
        .shift(-1)
        .fillna(0)
        .astype(float)
    )

    # Derive labels
    def _label(row):
        if row["wicket_next_over"] >= 1:
            return 0   # CRITICAL
        elif row["runs_next_over"] >= 8:
            return 1   # WARNING
        else:
            return 2   # INFO

    over_grp["label"] = over_grp.apply(_label, axis=1)

    # Rolling partnership (crude: 6 balls × over since last wicket)
    over_grp["balls_so_far"] = (over_grp["over"] + 1) * 6

    # Bowler type: we don't have per-over bowler in aggregated data
    # Use phase as proxy (powerplay = more pace, death = pace/mixed, middle = mixed)
    def _bowler_type(phase: str) -> str:
        if phase == "powerplay":
            return "pace"
        elif phase == "death":
            return "pace/mixed"
        return "mixed"

    over_grp["bowler_type"] = over_grp["phase"].apply(_bowler_type)
    over_grp["dew_active"]  = over_grp["over"] >= 14   # proxy: dew typical after over 14

    # Build text inputs
    rows = []
    for _, r in over_grp.iterrows():
        score   = int(r["innings_score"])
        wickets = int(r["innings_wickets"])
        over    = int(r["over"]) + 1
        phase   = str(r["phase"])
        dew     = "yes" if r["dew_active"] else "no"
        btype   = str(r["bowler_type"])
        # Simple partnership proxy: runs and balls since last wicket
        p_balls = int(r["balls_so_far"])
        p_runs  = int(r["runs_this_over"])  # current over as proxy

        text = (
            f"Over {over}. "
            f"Score {score}/{wickets}. "
            f"Partnership {p_runs} off {min(p_balls, 24)}. "
            f"Phase {phase}. "
            f"Dew active: {dew}. "
            f"Bowler: {btype}."
        )
        rows.append({"text": text, "label": int(r["label"])})

    df = pd.DataFrame(rows)
    # Downsample INFO class to 2× (WARNING + CRITICAL) to address class imbalance
    info_rows   = df[df["label"] == 2]
    other_rows  = df[df["label"] != 2]
    target_info = min(len(info_rows), len(other_rows) * 2)
    info_sample = info_rows.sample(n=target_info, random_state=42)
    df = pd.concat([other_rows, info_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------

def train(verbose: bool = True) -> None:
    try:
        from transformers import (
            DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
            Trainer, TrainingArguments,
        )
        import torch
        from torch.utils.data import Dataset
    except ImportError as e:
        print(f"[train_situation_classifier] Missing dependency: {e}")
        return

    if verbose:
        print("[train_situation_classifier] Building training data...")
    df = build_training_data()
    # Subsample for faster CPU training
    MAX_TRAIN_SAMPLES = 500
    if len(df) > MAX_TRAIN_SAMPLES:
        df = df.sample(n=MAX_TRAIN_SAMPLES, random_state=42).reset_index(drop=True)
    if verbose:
        print(f"[train_situation_classifier] {len(df)} training rows (capped at {MAX_TRAIN_SAMPLES})")
        print(f"  Label distribution: {df['label'].value_counts().to_dict()}")

    # Tokenise
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    encodings = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors="pt",
    )

    class _SituationDataset(Dataset):
        def __init__(self, enc, labels):
            self.input_ids      = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            self.labels         = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids":      self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels":         self.labels[idx],
            }

    labels  = df["label"].tolist()
    dataset = _SituationDataset(encodings, labels)

    n_val   = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    import torch
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3,
        id2label={0: "CRITICAL", 1: "WARNING", 2: "INFO"},
        label2id={"CRITICAL": 0, "WARNING": 1, "INFO": 2},
    )

    CLASSIFIER_PATH.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir              = str(CLASSIFIER_PATH),
        num_train_epochs        = 1,
        max_steps               = 200,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size  = 32,
        eval_strategy           = "no",
        save_strategy           = "no",
        logging_steps           = 50,
        learning_rate           = 2e-5,
        warmup_steps            = 20,
        weight_decay            = 0.01,
        use_cpu                 = not torch.cuda.is_available(),
        report_to               = "none",
        dataloader_num_workers  = 0,
    )

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
    )

    if verbose:
        print(f"[train_situation_classifier] Fine-tuning DistilBERT for 3 epochs...")

    trainer.train()

    model.save_pretrained(str(CLASSIFIER_PATH))
    tokenizer.save_pretrained(str(CLASSIFIER_PATH))
    if verbose:
        print(f"[train_situation_classifier] Saved -> {CLASSIFIER_PATH}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train(verbose=True)
    print("\n[train_situation_classifier] Done — situation_classifier/ ready for match_intelligence.py")
