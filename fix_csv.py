import csv
import os

path = r"c:\Users\Muhammad Wahib\Desktop\PSL DASHBOARD\psl_decision_engine\data\processed\player_index_2026_enriched.csv"
temp_path = r"c:\Users\Muhammad Wahib\Desktop\PSL DASHBOARD\psl_decision_engine\data\processed\player_index_2026_enriched_temp.csv"

# Known role corrections
role_fixes = {
    "Usman Khan": "WK-Batsman",
    "Usman Khan (MS)": "WK-Batsman",
    "Haseebullah Khan": "WK-Batsman",
    "Mohammad Haris": "WK-Batsman",
    "Arafat Minhas": "All-rounder",
    "Saim Ayub": "Batsman",
}

def clean_data():
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Clean roles and overseas flags
    for row in rows:
        name = row["player_name"].strip()
        nat = row["nationality"].strip().lower()
        
        # 1. Fix Overseas
        if nat == "pakistan":
            row["is_overseas"] = "False"
        else:
            row["is_overseas"] = "True"
            row["is_emerging"] = "False"  # Only Pakistanis can be emerging

        # 2. Fix Roles manually for obvious ones
        if name in role_fixes:
            row["primary_role"] = role_fixes[name]
            
        # 3. Dynamic heuristic: if Bowler but economy is 0.0 and bat SR > 100 -> Batsman
        role = row["primary_role"]
        eco = float(row["t20_career_economy"] or 0.0)
        bat_sr = float(row["t20_career_sr"] or 0.0)
        if role == "Bowler" and eco == 0.0 and bat_sr > 100.0:
            row["primary_role"] = "Batsman"

    with open(temp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    os.replace(temp_path, path)
    print("Cleaned player_index_2026_enriched.csv successfully.")

if __name__ == "__main__":
    clean_data()
