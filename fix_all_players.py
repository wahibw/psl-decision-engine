import csv
import os

path = r"c:\Users\Muhammad Wahib\Desktop\PSL DASHBOARD\psl_decision_engine\data\processed\player_index_2026_enriched.csv"
temp_path = r"c:\Users\Muhammad Wahib\Desktop\PSL DASHBOARD\psl_decision_engine\data\processed\player_index_2026_enriched_temp2.csv"

# Comprehensive Override Dictionary
# (Overrides apply only to the specified keys)
overrides = {
    "Abdul Samad": {"nationality": "Pakistan", "primary_role": "Batsman", "bowling_style": "Right-arm offbreak"},
    "Usman Tariq": {"primary_role": "Bowler"},
    "Mustafizur Rahman": {"batting_style": "Left-hand bat"},
    "Hussain Talat": {"bowling_style": "Right-arm medium"},
    "Parvez Hussain Emon": {"batting_style": "Left-hand bat"},
    "Haseebullah Khan": {"batting_style": "Left-hand bat"},
    "Sufyan Moqim": {"nationality": "Pakistan", "batting_style": "Left-hand bat", "bowling_style": "Left-arm wrist-spin"},
    "Ali Raza": {"primary_role": "Bowler", "batting_style": "Right-hand bat", "bowling_style": "Right-arm fast"},
    "Michael Bracewell": {"batting_style": "Left-hand bat"},
    "Yasir Khan": {"primary_role": "Batsman"},
    "Amad Butt": {"bowling_style": "Right-arm fast"},
    "Asif Afridi": {"batting_style": "Left-hand bat", "bowling_style": "Slow left-arm orthodox"},
    "Kamran Ghulam": {"bowling_style": "Slow left-arm orthodox"},
    "Mir Hamza": {"batting_style": "Left-hand bat"},
    "Saad Baig": {"primary_role": "WK-Batsman"},
    "Faheem Ashraf": {"batting_style": "Left-hand bat"},
    "Andries Gous": {"nationality": "USA"},
    "Mehran Mumtaz": {"batting_style": "Left-hand bat", "bowling_style": "Slow left-arm orthodox"},
    "Jahandad Khan": {"batting_style": "Left-hand bat", "bowling_style": "Left-arm fast-medium"},
    "Bismillah Khan": {"batting_style": "Right-hand bat"},
    "Hasan Nawaz": {"primary_role": "Batsman"},
    "Shamyl Hussain": {"batting_style": "Left-hand bat", "primary_role": "Batsman"},
    "Faisal Akram": {"batting_style": "Left-hand bat", "bowling_style": "Left-arm wrist-spin", "primary_role": "Bowler"},
    "Arafat Minhas": {"batting_style": "Left-hand bat", "bowling_style": "Slow left-arm orthodox", "primary_role": "All-rounder"},
    "Bevon Jacobs": {"nationality": "South Africa", "primary_role": "Batsman"},
    "Akif Javed": {"bowling_style": "Left-arm fast-medium"},
    "Muhammad Irfan Khan": {"batting_style": "Right-hand bat"},
    "Hassan Khan": {"bowling_style": "Slow left-arm orthodox"},
    "Shayan Jahangir": {"nationality": "USA"},
    "Hammad Azam": {"bowling_style": "Right-arm medium"},
    "Maaz Sadaqat": {"batting_style": "Left-hand bat", "primary_role": "All-rounder", "bowling_style": "Slow left-arm orthodox"},
    "Saad Ali": {"batting_style": "Left-hand bat", "primary_role": "Batsman"},
    "Salman Mirza": {"bowling_style": "Left-arm fast-medium"},
    "Saad Masood": {"primary_role": "All-rounder", "bowling_style": "Right-arm medium"},
    "Lachlan Shaw": {"primary_role": "WK-Batsman"},
    "Dunith Wellalage": {"nationality": "Sri Lanka"},
    "Rubin Hermann": {"nationality": "South Africa"},
    "Alzarri Joseph": {"nationality": "West Indies"},
    "Shoriful Islam": {"nationality": "Bangladesh"},
    "Glenn Maxwell": {"nationality": "Australia"},
}

def clean_data():
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Clean roles and overseas flags
    for row in rows:
        name = row["player_name"].strip()
        
        # Apply specific overrides
        if name in overrides:
            for k, v in overrides[name].items():
                if k in fieldnames:
                    row[k] = v

        # Recalculate is_overseas dynamically based on fixed nationality
        nat = row["nationality"].strip().lower()
        if nat == "pakistan":
            row["is_overseas"] = "False"
        else:
            row["is_overseas"] = "True"
            row["is_emerging"] = "False"  # Non-Pakistanis cannot be emerging

    with open(temp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    os.replace(temp_path, path)
    print("Surgically patched 40+ records in player_index_2026_enriched.csv successfully.")
    
    # Audit check
    for row in rows:
        if row["player_name"].strip() == "Abdul Samad":
            print(f"Audit {row['player_name']}: NAT={row['nationality']}, ROLE={row['primary_role']}, OS={row['is_overseas']}")
        if row["player_name"].strip() == "Usman Tariq":
            print(f"Audit {row['player_name']}: ROLE={row['primary_role']}")

if __name__ == "__main__":
    clean_data()
