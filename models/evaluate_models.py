# models/evaluate_models.py
# Prints an accuracy / quality report for all three saved models.
#
# Run:  python models/evaluate_models.py

from __future__ import annotations

import pickle
import sys
from pathlib import Path

# Ensure project root is on path so "models.train_*" imports resolve
PROJ_ROOT  = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJ_ROOT / "models" / "saved"
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))


def _load(name: str) -> dict | None:
    path = MODELS_DIR / name
    if not path.exists():
        print(f"  [MISSING] {name} — run the training script first")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate_bowling_model(payload: dict) -> None:
    meta = payload["meta"]
    print(f"\n  bowling_model.pkl")
    print(f"  {'-'*50}")
    print(f"  Task           : Predict runs per over (regression)")
    print(f"  Bowlers        : {meta['n_bowlers']}")
    print(f"  Train rows     : {meta['train_rows']:,} overs")
    print(f"  Test rows      : {meta['test_rows']:,} overs")
    print(f"  Train seasons  : {[int(s) for s in meta['train_seasons']]}")
    print(f"  Test seasons   : {[int(s) for s in meta['test_seasons']]}")
    print(f"  Train RMSE     : {meta['rmse_train']:.3f} runs/over")
    print(f"  Test  RMSE     : {meta['rmse_test']:.3f} runs/over")
    print(f"  Test  MAE      : {meta['mae_test']:.3f} runs/over")
    print(f"  Runs cap       : {meta['runs_cap']}")

    # Sanity: spot-check top and bottom bowlers at death
    from models.train_bowling import score_bowler
    context = dict(phase="death", over_num=18, innings=2, score_rate=10.0, wickets_lost=3)
    preds = []
    for b in list(payload["bowler_lookup"].keys())[:30]:
        sc = score_bowler(b, payload=payload, **context)
        preds.append((b, sc))
    preds.sort(key=lambda x: x[1])
    print(f"\n  Best 5 death bowlers (model prediction):")
    for b, sc in preds[:5]:
        print(f"    {b:<30}  {sc:.2f} runs/over")
    print(f"  Worst 5 death bowlers:")
    for b, sc in preds[-5:]:
        print(f"    {b:<30}  {sc:.2f} runs/over")


def evaluate_xi_scorer(payload: dict) -> None:
    meta = payload["meta"]
    print(f"\n  xi_scorer.pkl")
    print(f"  {'-'*50}")
    print(f"  Task           : Predict player match impact (0-100)")
    print(f"  Players        : {meta['n_players']}")
    print(f"  Train rows     : {meta['train_rows']:,} player-matches")
    print(f"  Test rows      : {meta['test_rows']:,} player-matches")
    print(f"  Train seasons  : {[int(s) for s in meta['train_seasons'] if str(s) != 'nan']}")
    print(f"  Test seasons   : {[int(s) for s in meta['test_seasons'] if str(s) != 'nan']}")
    print(f"  Train RMSE     : {meta['rmse_train']:.2f} / 100")
    print(f"  Test  RMSE     : {meta['rmse_test']:.2f} / 100")
    print(f"  Test  MAE      : {meta['mae_test']:.2f} / 100")

    # Score a reference squad at Lahore, no weather modifier
    from models.train_xi_scorer import score_player
    ref_squad = [
        "Babar Azam", "Fakhar Zaman", "Mohammad Rizwan",
        "Shaheen Shah Afridi", "Shadab Khan", "Imad Wasim",
        "Hasan Ali", "Mohammad Nawaz",
    ]
    print(f"\n  Reference squad scores (Gaddafi Stadium, Lahore, innings 1):")
    print(f"  {'Player':<28}  Base   Dew(spin -40%)")
    print(f"  {'-'*55}")
    for p in ref_squad:
        base = score_player(p, "Gaddafi Stadium, Lahore", 1, payload)
        dew  = score_player(p, "Gaddafi Stadium, Lahore", 1, payload, spinner_penalty=0.6)
        diff = f"-> {dew:.1f}" if dew != base else "     --"
        print(f"  {p:<28}  {base:>5.1f}  {diff}")


def evaluate_scenario_model(payload: dict) -> None:
    from models.train_batting_scenarios import rank_players_for_scenario, SCENARIO_NAMES
    print(f"\n  batting_scenario_model.pkl")
    print(f"  {'-'*50}")
    for sc, name in SCENARIO_NAMES.items():
        df = payload["scenario_lookup"][sc]
        print(f"  Scenario {sc} ({name}): {len(df)} qualifying batters")

    # Test ranking consistency
    test_players = ["Babar Azam", "Shaheen Shah Afridi", "Imad Wasim", "Fakhar Zaman", "Khushdil Shah"]
    print(f"\n  Scenario ranking check ({', '.join(test_players)}):")
    for sc in ["A", "B", "C", "D"]:
        ranked = rank_players_for_scenario(test_players, sc, payload)
        top2 = " > ".join([f"{p.split()[-1]}({s:.0f})" for p, s in ranked[:3]])
        print(f"    Scenario {sc}: {top2}")


def main() -> None:
    print(f"\n{'='*55}")
    print(f"  PSL Decision Engine — Model Evaluation Report")
    print(f"{'='*55}")

    bowling  = _load("bowling_model.pkl")
    xi       = _load("xi_scorer.pkl")
    scenario = _load("batting_scenario_model.pkl")

    if bowling:
        evaluate_bowling_model(bowling)
    if xi:
        evaluate_xi_scorer(xi)
    if scenario:
        evaluate_scenario_model(scenario)

    print(f"\n{'='*55}")
    print(f"  All models evaluated.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
