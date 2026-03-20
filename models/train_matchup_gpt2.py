# models/train_matchup_gpt2.py
# Fine-tunes GPT-2 (small) to generate natural-language scouting notes
# from H2H matchup statistics.
#
# Training data: matchup_matrix.parquet pairs with balls >= 8
# Prompt: "Bowler: X. Batter: Y. Balls: N. SR: S. Dismissals: D. Economy: E. Scouting note:"
# Completion: 1-sentence note (template-generated for cold-start training)
#
# Save: models/saved/matchup_gpt2/
# Run:  python models/train_matchup_gpt2.py

from __future__ import annotations

import sys
from pathlib import Path

PROJ_ROOT  = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJ_ROOT / "models" / "saved"
DATA_DIR   = PROJ_ROOT / "data" / "processed"
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

GPT2_SAVE_PATH = MODELS_DIR / "matchup_gpt2"
MIN_BALLS      = 8    # minimum H2H balls to generate a meaningful note
MAX_NOTE_TOKENS= 60   # max tokens in generated note


# ---------------------------------------------------------------------------
# NOTE TEMPLATE (cold-start: generates training completions from stats)
# ---------------------------------------------------------------------------

def _template_note(row) -> str:
    """
    Generate a one-sentence scouting note from H2H stats.
    This provides supervision signal for the initial fine-tune.
    """
    batter     = str(row["batter"])
    bowler     = str(row["bowler"])
    balls      = int(row["balls"])
    sr         = float(row["sr"])
    dismissals = int(row["dismissals"])
    economy    = float(row["runs"]) / balls * 6 if balls > 0 else 8.0
    adv        = float(row["bowler_adv"])
    b_last     = batter.split()[-1]
    p_last     = bowler.split()[-1]

    if adv >= 20 and dismissals >= 2:
        return (
            f"{p_last} dominates {b_last}: {dismissals} dismissals in {balls} balls "
            f"(SR {sr:.0f}), making {p_last} the ideal bowler when {b_last} is at the crease."
        )
    elif adv >= 10 and dismissals >= 1:
        return (
            f"{p_last} holds the edge over {b_last} with {dismissals} dismissal(s) "
            f"in {balls} balls and economy of {economy:.1f}."
        )
    elif adv <= -20:
        return (
            f"{b_last} has a strong advantage over {p_last}: SR {sr:.0f} in {balls} balls "
            f"— consider protecting {p_last} from this matchup."
        )
    elif adv <= -10:
        return (
            f"{b_last} is comfortable against {p_last}, scoring at SR {sr:.0f} "
            f"in {balls} balls — use {p_last} sparingly vs {b_last}."
        )
    elif dismissals == 0:
        return (
            f"{b_last} has never been dismissed by {p_last} in {balls} PSL balls "
            f"(SR {sr:.0f}) — proceed with caution."
        )
    else:
        return (
            f"{p_last} vs {b_last}: neutral matchup over {balls} balls "
            f"(SR {sr:.0f}, {dismissals} dismissal(s)); outcome depends on conditions."
        )


# ---------------------------------------------------------------------------
# BUILD DATASET
# ---------------------------------------------------------------------------

def build_training_texts(df) -> list[str]:
    """
    Build list of prompt+completion strings for causal LM fine-tuning.
    GPT-2 uses teacher-forcing on the full sequence (prompt + note).
    """
    texts = []
    for _, row in df.iterrows():
        balls   = int(row["balls"])
        economy = float(row["runs"]) / balls * 6 if balls > 0 else 8.0
        prompt = (
            f"Bowler: {row['bowler']}. "
            f"Batter: {row['batter']}. "
            f"Balls: {balls}. "
            f"SR: {row['sr']:.0f}. "
            f"Dismissals: {int(row['dismissals'])}. "
            f"Economy: {economy:.1f}. "
            f"Scouting note: "
        )
        completion = _template_note(row)
        texts.append(prompt + completion + "<|endoftext|>")
    return texts


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------

def train(verbose: bool = True) -> None:
    try:
        from transformers import (
            GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments,
            DataCollatorForLanguageModeling,
        )
        import torch
        import pandas as pd
        from torch.utils.data import Dataset
    except ImportError as e:
        print(f"[train_matchup_gpt2] Missing dependency: {e}")
        return

    matrix_path = DATA_DIR / "matchup_matrix.parquet"
    if not matrix_path.exists():
        print(f"[train_matchup_gpt2] matchup_matrix.parquet not found")
        return

    df = pd.read_parquet(matrix_path)
    df = df[df["balls"] >= MIN_BALLS].reset_index(drop=True)
    # Subsample for faster CPU training
    MAX_TRAIN_SAMPLES = 500
    if len(df) > MAX_TRAIN_SAMPLES:
        df = df.sample(n=MAX_TRAIN_SAMPLES, random_state=42).reset_index(drop=True)
    if verbose:
        print(f"[train_matchup_gpt2] {len(df)} matchups (capped at {MAX_TRAIN_SAMPLES})")

    texts = build_training_texts(df)
    if verbose:
        print(f"[train_matchup_gpt2] {len(texts)} training examples built")
        print(f"  Sample: {texts[0][:120]}...")

    # Tokenizer + model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenise all texts
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="pt",
    )

    class _MatchupDataset(Dataset):
        def __init__(self, enc):
            self.input_ids      = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                "input_ids":      self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels":         self.input_ids[idx].clone(),
            }

    dataset = _MatchupDataset(encodings)

    # Split 90/10
    n_val    = max(1, len(dataset) // 10)
    n_train  = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    GPT2_SAVE_PATH.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir            = str(GPT2_SAVE_PATH),
        num_train_epochs      = 1,
        max_steps             = 200,
        per_device_train_batch_size = 8,
        per_device_eval_batch_size  = 8,
        eval_strategy         = "no",
        save_strategy         = "no",
        logging_steps         = 50,
        learning_rate         = 5e-5,
        warmup_steps          = 20,
        weight_decay          = 0.01,
        use_cpu               = not torch.cuda.is_available(),
        report_to             = "none",
        dataloader_num_workers= 0,
    )

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    if verbose:
        print(f"[train_matchup_gpt2] Fine-tuning GPT-2 for 3 epochs on {device}...")

    trainer.train()

    # Save model + tokenizer
    model.save_pretrained(str(GPT2_SAVE_PATH))
    tokenizer.save_pretrained(str(GPT2_SAVE_PATH))
    if verbose:
        print(f"[train_matchup_gpt2] Saved -> {GPT2_SAVE_PATH}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train(verbose=True)
    print("\n[train_matchup_gpt2] Done. matchup_gpt2/ is ready for use in matchup_engine.py")
