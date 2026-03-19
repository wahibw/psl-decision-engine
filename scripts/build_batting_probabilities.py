"""
scripts/build_batting_probabilities.py

Reads data/processed/opposition_profiles.csv (season-level rows),
counts per-season batting position occupancy for every player × team,
and writes data/processed/batting_order_probabilities.json.

Output schema:
{
  "<team>": {
    "<player_name>": {
      "most_common_position": int,
      "position_range":       "min-max" | "n",   # e.g. "1-3" or "2"
      "seasons_count":        int,                # PSL seasons in data
      "position_confidence":  "High" | "Medium" | "Low",
      "position_probs":       {"1": float, ...}   # position -> fraction of seasons
    }
  }
}

Confidence rules:
  High   — player appears in 3+ seasons at this team
  Medium — 2 seasons
  Low    — 1 season  (shown as "Single season — position uncertain")
"""

from __future__ import annotations

import json
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJ_ROOT   = Path(__file__).resolve().parent.parent
INPUT_CSV   = PROJ_ROOT / "data" / "processed" / "opposition_profiles.csv"
OUTPUT_JSON = PROJ_ROOT / "data" / "processed" / "batting_order_probabilities.json"


def _load_batting_orders(csv_path: Path) -> dict:
    """
    Returns nested dict:
      { team -> { player_name -> [(season, position), ...] } }

    Stores (season, position) tuples so build() can count distinct
    PSL seasons rather than raw position appearances.
    Skips the season=0 aggregate row.
    """
    result: dict[str, dict[str, list[tuple]]] = defaultdict(lambda: defaultdict(list))

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            team   = row.get("team", "").strip()
            season = row.get("season", "0").strip()
            order_raw = row.get("typical_batting_order", "").strip()

            if not team or not order_raw or season == "0":
                continue

            try:
                order: dict = json.loads(order_raw)
            except (json.JSONDecodeError, ValueError):
                continue

            for pos_str, player in order.items():
                try:
                    pos = int(pos_str)
                except ValueError:
                    continue
                player = str(player).strip()
                if player:
                    result[team][player].append((season, pos))

    return result


def _build_probabilities(positions: list[int]) -> dict[str, float]:
    """Given a list of positions seen, return a {pos_str: fraction} dict."""
    total = len(positions)
    counts: dict[int, int] = defaultdict(int)
    for p in positions:
        counts[p] += 1
    return {str(pos): round(cnt / total, 4) for pos, cnt in sorted(counts.items())}


def _position_range(positions: list[int]) -> str:
    lo, hi = min(positions), max(positions)
    return str(lo) if lo == hi else f"{lo}-{hi}"


def _confidence(seasons_count: int) -> str:
    if seasons_count >= 3:
        return "High"
    if seasons_count == 2:
        return "Medium"
    return "Low"


def build(csv_path: Path = INPUT_CSV, output_path: Path = OUTPUT_JSON) -> dict:
    raw = _load_batting_orders(csv_path)

    output: dict = {}
    for team, players in sorted(raw.items()):
        output[team] = {}
        for player, season_pos_list in sorted(players.items()):
            seasons_count  = len(set(s for s, _ in season_pos_list))
            positions_only = [p for _, p in season_pos_list]
            probs   = _build_probabilities(positions_only)
            output[team][player] = {
                "most_common_position": int(max(probs, key=lambda k: probs[k])),
                "position_range":       _position_range(positions_only),
                "seasons_count":        seasons_count,
                "position_confidence":  _confidence(seasons_count),
                "position_probs":       probs,
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    return output


if __name__ == "__main__":
    print(f"Reading  : {INPUT_CSV}")
    print(f"Writing  : {OUTPUT_JSON}")

    result = build()

    # Summary stats
    total_teams   = len(result)
    total_players = sum(len(v) for v in result.values())
    high   = sum(1 for t in result.values() for p in t.values() if p["position_confidence"] == "High")
    medium = sum(1 for t in result.values() for p in t.values() if p["position_confidence"] == "Medium")
    low    = sum(1 for t in result.values() for p in t.values() if p["position_confidence"] == "Low")

    coverage = sum(1 for t in result.values() for p in t.values() if p["seasons_count"] >= 2)

    print(f"\nBuilt batting probabilities for {total_teams} teams, {total_players} players")
    print(f"Coverage: {coverage} player-position pairs with >= 2 season observations")
    print(f"  High confidence  : {high}")
    print(f"  Medium confidence: {medium}")
    print(f"  Low confidence   : {low}")

    # Spot-check: print first team's first 3 players
    first_team = next(iter(result))
    print(f"\nSpot-check — {first_team}:")
    for i, (player, data) in enumerate(result[first_team].items()):
        if i >= 3:
            break
        _conf_label = (
            "Single season — position uncertain"
            if data["position_confidence"] == "Low"
            else f"{data['position_confidence']} confidence"
        )
        print(f"  {player}: pos {data['most_common_position']} ({data['position_range']}), "
              f"{data['seasons_count']} season(s), {_conf_label}")

    sys.exit(0)
