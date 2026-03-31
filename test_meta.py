import sys
from pathlib import Path

sys.path.append(r"c:\Users\Muhammad Wahib\Desktop\PSL DASHBOARD\psl_decision_engine")

from engine.xi_selector import _load_meta

path = r"c:\Users\Muhammad Wahib\Desktop\PSL DASHBOARD\psl_decision_engine\data\processed\player_index_2026_enriched.csv"
meta = _load_meta(path)

print("Alzarri Joseph:", meta.get("Alzarri Joseph", {}))
print("Glenn Maxwell:", meta.get("Glenn Maxwell", {}))
print("Shaheen Shah Afridi:", meta.get("Shaheen Shah Afridi", {}))
