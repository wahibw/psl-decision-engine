"""
scripts/fill_bowl_dot_pct.py
Fill bowl_dot_pct in player_index_2026_enriched.csv using ball_by_ball.parquet.

Logic:
  - Legal delivery  = is_wide==False AND is_noball==False
  - Dot ball        = legal delivery where runs_batter==0 AND runs_extras==0
  - bowl_dot_pct    = dots / legal_balls * 100  (min 60 legal balls threshold)
  - Only fills NaN values — never overwrites existing data
  - Name matching   = exact first, then difflib fuzzy cutoff=0.85
  - Output          = player_index_2026_enriched_v2.csv + bowl_dot_pct_source column
"""

import difflib
import pandas as pd

PARQUET    = "data/processed/ball_by_ball.parquet"
INDEX_IN   = "data/processed/player_index_2026_enriched.csv"
INDEX_OUT  = "data/processed/player_index_2026_enriched_v2.csv"
MIN_BALLS  = 60
FUZZY_CUT  = 0.85

# ---------------------------------------------------------------------------
# 1. Compute bowl_dot_pct per bowler from parquet
# ---------------------------------------------------------------------------
bbb = pd.read_parquet(PARQUET)

legal = bbb[(bbb["is_wide"] == False) & (bbb["is_noball"] == False)].copy()
legal["is_dot"] = (legal["runs_batter"] == 0) & (legal["runs_extras"] == 0)

agg = (
    legal.groupby("bowler")
    .agg(legal_balls=("is_dot", "count"), dots=("is_dot", "sum"))
    .reset_index()
)
agg = agg[agg["legal_balls"] >= MIN_BALLS].copy()
agg["computed_dot_pct"] = (agg["dots"] / agg["legal_balls"] * 100).round(1)

cricsheet_names = set(agg["bowler"].tolist())

# ---------------------------------------------------------------------------
# 2. Load player index
# ---------------------------------------------------------------------------
idx = pd.read_csv(INDEX_IN)

# Ensure required columns exist before any modifications
if "bowl_dot_pct_source" not in idx.columns:
    idx["bowl_dot_pct_source"] = ""
if "is_emerging" not in idx.columns:
    # Derive from existing data rather than losing the column
    idx["is_emerging"] = (
        (idx.get("nationality", "Pakistan") == "Pakistan") &
        (idx.get("is_overseas", False) == False) &
        (idx.get("psl_seasons_played", 99) <= 2)
    )

# Mark existing data source
idx.loc[idx["bowl_dot_pct"].notna() & (idx["bowl_dot_pct_source"] == ""), "bowl_dot_pct_source"] = "pre_existing"

existing_count = idx["bowl_dot_pct"].notna().sum()

# ---------------------------------------------------------------------------
# 3. Name matching: exact then fuzzy
# ---------------------------------------------------------------------------
filled       = 0
fuzzy_used   = []
no_match     = []

for i, row in idx.iterrows():
    if pd.notna(row["bowl_dot_pct"]):
        continue  # already has data — skip

    pname = row["player_name"]

    # --- exact match ---
    match = agg[agg["bowler"] == pname]
    if not match.empty:
        idx.at[i, "bowl_dot_pct"]        = match.iloc[0]["computed_dot_pct"]
        idx.at[i, "bowl_dot_pct_source"] = "psl_ballbyball"
        filled += 1
        continue

    # --- fuzzy match ---
    candidates = difflib.get_close_matches(pname, cricsheet_names, n=1, cutoff=FUZZY_CUT)
    if candidates:
        best = candidates[0]
        match = agg[agg["bowler"] == best]
        idx.at[i, "bowl_dot_pct"]        = match.iloc[0]["computed_dot_pct"]
        idx.at[i, "bowl_dot_pct_source"] = "psl_ballbyball"
        fuzzy_used.append((pname, best, match.iloc[0]["computed_dot_pct"]))
        filled += 1
    else:
        no_match.append(pname)

# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------
idx.to_csv(INDEX_OUT, index=False)

# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------
still_null = idx["bowl_dot_pct"].isna().sum()

print("=" * 60)
print("bowl_dot_pct fill summary")
print("=" * 60)
print(f"  Pre-existing values  : {existing_count}")
print(f"  Newly filled         : {filled}")
print(f"  Still null           : {still_null}")
print(f"  Total active rows    : {len(idx)}")
print()

if fuzzy_used:
    print(f"Fuzzy matches used ({len(fuzzy_used)}):")
    for player, matched, pct in sorted(fuzzy_used):
        print(f"  '{player}'  ->  '{matched}'  ({pct}%)")
    print()

if no_match:
    # Filter to players who actually bowl (not pure batters)
    bowling_roles = {"Bowler", "All-rounder", "Batting All-rounder", "Bowling All-rounder"}
    no_match_bowlers = [
        n for n in no_match
        if idx.loc[idx["player_name"] == n, "primary_role"].values[0] in bowling_roles
    ]
    if no_match_bowlers:
        print(f"No CricSheet match for bowlers ({len(no_match_bowlers)}):")
        for n in sorted(no_match_bowlers):
            role = idx.loc[idx["player_name"] == n, "primary_role"].values[0]
            team = idx.loc[idx["player_name"] == n, "current_team_2026"].values[0]
            print(f"  {n:<30} {role:<22} {team}")
        print()

print(f"Output written to: {INDEX_OUT}")
