"""scripts/update_squads_2026.py — PSL 2026 squad update."""
import pandas as pd

CSV_PATH = "data/processed/player_index_2026_enriched.csv"
df = pd.read_csv(CSV_PATH, dtype=str)

def set_player(df, csv_name, team, overseas):
    mask = df["player_name"] == csv_name
    if mask.sum() == 0:
        return False
    df.loc[mask, "current_team_2026"] = team
    df.loc[mask, "is_overseas"]       = str(overseas)
    df.loc[mask, "is_active"]         = "True"
    return True

# ── SQUAD DEFINITIONS  (csv_name, team, overseas) ────────────────────────────
SQUADS = [
    # KARACHI KINGS
    ("David Warner",            "Karachi Kings", True),
    ("Hasan Ali",               "Karachi Kings", False),
    ("Mohammad Abbas Afridi",   "Karachi Kings", False),
    ("Khushdil Shah",           "Karachi Kings", False),
    ("Saad Baig",               "Karachi Kings", False),
    ("Moeen Ali",               "Karachi Kings", True),
    ("Azam Khan",               "Karachi Kings", False),
    ("Salman Ali Agha",         "Karachi Kings", False),
    ("Shahid Aziz",             "Karachi Kings", False),
    ("Mir Hamza",               "Karachi Kings", False),
    ("Adam Zampa",              "Karachi Kings", True),
    ("Hamza Sohail",            "Karachi Kings", False),
    ("Aqib Ilyas",              "Karachi Kings", False),
    ("Khuzaima Bin Tanveer",    "Karachi Kings", True),
    ("Johnson Charles",         "Karachi Kings", True),
    ("Muhammad Waseem",         "Karachi Kings", True),
    ("Ihsanullah",              "Karachi Kings", False),
    ("Rizwanullah",             "Karachi Kings", False),
    # LAHORE QALANDARS
    ("Shaheen Shah Afridi",     "Lahore Qalandars", False),
    ("Abdullah Shafique",       "Lahore Qalandars", False),
    ("Sikandar Raza",           "Lahore Qalandars", True),
    ("Mohammad Naeem",          "Lahore Qalandars", False),
    ("Mustafizur Rahman",       "Lahore Qalandars", True),
    ("Haris Rauf",              "Lahore Qalandars", False),
    ("Usama Mir",               "Lahore Qalandars", False),
    ("Fakhar Zaman",            "Lahore Qalandars", False),
    ("Ubaid Shah",              "Lahore Qalandars", False),
    ("Haseebullah Khan",        "Lahore Qalandars", False),
    ("Mohammad Farooq",         "Lahore Qalandars", False),
    ("Dasun Shanaka",           "Lahore Qalandars", True),
    ("Parvez Hussain Emon",     "Lahore Qalandars", True),
    ("Asif Ali",                "Lahore Qalandars", False),
    ("Hussain Talat",           "Lahore Qalandars", False),
    ("Tayyab Tahir",            "Lahore Qalandars", False),
    # QUETTA GLADIATORS
    ("Saud Shakeel",            "Quetta Gladiators", False),
    ("Usman Tariq",             "Quetta Gladiators", False),
    ("Hasan Nawaz",             "Quetta Gladiators", False),
    ("Shamyl Hussain",          "Quetta Gladiators", False),
    ("Rilee Rossouw",           "Quetta Gladiators", True),
    ("Ahmad Daniyal",           "Quetta Gladiators", False),
    ("Jahanzaib Sultan",        "Quetta Gladiators", False),
    ("Jahandad Khan",           "Quetta Gladiators", False),
    ("Khawaja Mohammad Nafay",  "Quetta Gladiators", False),
    ("Wasim Akram Jnr",         "Quetta Gladiators", False),
    ("Khan Zaib",               "Quetta Gladiators", False),
    ("Bismillah Khan",          "Quetta Gladiators", False),
    ("Brett Hampton",           "Quetta Gladiators", True),
    ("Sam Harper",              "Quetta Gladiators", True),
    ("Bevon Jacobs",            "Quetta Gladiators", True),
    ("Abrar Ahmed",             "Quetta Gladiators", False),
    ("Ben McDermott",           "Quetta Gladiators", True),
    ("Tom Curran",              "Quetta Gladiators", True),
    # MULTAN SULTANS
    ("Ashton Turner",           "Multan Sultans", True),
    ("Mohammad Nawaz",          "Multan Sultans", False),
    ("Faisal Akram",            "Multan Sultans", False),
    ("Arafat Minhas",           "Multan Sultans", False),
    ("Sahibzada Farhan",        "Multan Sultans", False),
    ("Steve Smith",             "Multan Sultans", True),
    ("Peter Siddle",            "Multan Sultans", True),
    ("Tabraiz Shamsi",          "Multan Sultans", True),
    ("Lachlan Shaw",            "Multan Sultans", True),
    ("Delano Potgieter",        "Multan Sultans", True),
    ("Josh Philippe",           "Multan Sultans", True),
    ("Shan Masood",             "Multan Sultans", False),
    ("Momin Qamar",             "Multan Sultans", False),
    ("Muhammad Awais Zafar",    "Multan Sultans", False),
    ("Arshad Iqbal",            "Multan Sultans", False),
    ("Mohammad Wasim Jnr",      "Multan Sultans", False),
    # RAWALPINDIZ
    ("Mohammad Rizwan",         "Rawalpindiz", False),
    ("Sam Billings",            "Rawalpindiz", True),
    ("Yasir Khan",              "Rawalpindiz", False),
    ("Naseem Shah",             "Rawalpindiz", False),
    ("Rishad Hossain",          "Rawalpindiz", True),
    ("Daryl Mitchell",          "Rawalpindiz", True),
    ("Mohammad Amir",           "Rawalpindiz", False),
    ("Abdullah Fazal",          "Rawalpindiz", False),
    ("Amad Butt",               "Rawalpindiz", False),
    ("Dian Forrestor",          "Rawalpindiz", True),
    ("Laurie Evans",            "Rawalpindiz", True),
    ("Asif Afridi",             "Rawalpindiz", False),
    ("Kamran Ghulam",           "Rawalpindiz", False),
    ("Fawad Ali",               "Rawalpindiz", False),
    ("Mohammad Amir Khan",      "Rawalpindiz", False),
    ("Shahzaib Khan",           "Rawalpindiz", False),
    ("Jake Fraser-McGurk",      "Rawalpindiz", True),
    ("Saad Masood",             "Rawalpindiz", False),
    # PESHAWAR ZALMI
    ("Babar Azam",              "Peshawar Zalmi", False),
    ("Sufyan Moqim",            "Peshawar Zalmi", False),
    ("Abdul Samad",             "Peshawar Zalmi", False),
    ("Ali Raza",                "Peshawar Zalmi", False),
    ("Aaron Hardie",            "Peshawar Zalmi", True),
    ("Aamir Jamal",             "Peshawar Zalmi", False),
    ("Khuram Shahzad",          "Peshawar Zalmi", False),
    ("Mohammad Haris",          "Peshawar Zalmi", False),
    ("Khalid Usman",            "Peshawar Zalmi", False),
    ("Abdul Subhan",            "Peshawar Zalmi", False),
    ("James Vince",             "Peshawar Zalmi", True),
    ("Michael Bracewell",       "Peshawar Zalmi", True),
    ("Kusal Mendis",            "Peshawar Zalmi", True),
    ("Iftikhar Ahmed",          "Peshawar Zalmi", False),
    ("Mirza Tahir Baig",        "Peshawar Zalmi", False),
    ("Kashif Ali",              "Peshawar Zalmi", False),
    # HYDERABAD KINGSMEN
    ("Marnus Labuschagne",      "Hyderabad Kingsmen", True),
    ("Usman Khan",              "Hyderabad Kingsmen", False),
    ("Akif Javed",              "Hyderabad Kingsmen", False),
    ("Maaz Sadaqat",            "Hyderabad Kingsmen", False),
    ("Saim Ayub",               "Hyderabad Kingsmen", False),
    ("Mohammad Ali",            "Hyderabad Kingsmen", False),
    ("Kusal Perera",            "Hyderabad Kingsmen", True),
    ("Muhammad Irfan Khan",     "Hyderabad Kingsmen", False),
    ("Hassan Khan",             "Hyderabad Kingsmen", True),
    ("Shayan Jahangir",         "Hyderabad Kingsmen", True),
    ("Hammad Azam",             "Hyderabad Kingsmen", False),
    ("Riley Meredith",          "Hyderabad Kingsmen", True),
    ("Sharjeel Khan",           "Hyderabad Kingsmen", False),
    ("Asif Mehmood",            "Hyderabad Kingsmen", False),
    ("Hunain Shah",             "Hyderabad Kingsmen", False),
    ("Rizwan Mehmood",          "Hyderabad Kingsmen", False),
    ("Saad Ali",                "Hyderabad Kingsmen", False),
    ("Tayyab Arif",             "Hyderabad Kingsmen", False),
    # ISLAMABAD UNITED
    ("Shadab Khan",             "Islamabad United", False),
    ("Salman Irshad",           "Islamabad United", False),
    ("Andries Gous",            "Islamabad United", True),
    ("Devon Conway",            "Islamabad United", True),
    ("Faheem Ashraf",           "Islamabad United", False),
    ("Mehran Mumtaz",           "Islamabad United", False),
    ("Max Bryant",              "Islamabad United", True),
    ("Mark Chapman",            "Islamabad United", True),
    ("Mir Hamza Sajjad",        "Islamabad United", False),
    ("Sameen Gul",              "Islamabad United", False),
    ("Sameer Minhas",           "Islamabad United", False),
    ("Imad Wasim",              "Islamabad United", False),
    ("Richard Gleeson",         "Islamabad United", True),
    ("Haider Ali",              "Islamabad United", False),
    ("Mohammad Hasnain",        "Islamabad United", False),
    ("Dipendra Singh Airee",    "Islamabad United", True),
    ("Salman Mirza",            "Islamabad United", False),
]

# ── NEW ROWS  (name, team, overseas, role, bat_style, bowl_style, sr, eco) ───
NEW_ROWS = [
    # Lahore Qalandars
    ("Dunith Wellalage",   "Lahore Qalandars",   True,  "All-rounder", "Left-hand bat",  "Slow left-arm orthodox", 118.0, 7.2),
    ("Rubin Hermann",      "Lahore Qalandars",   True,  "Bowler",      "Left-hand bat",  "Left-arm fast-medium",   None,  8.5),
    ("Maaz Khan",          "Lahore Qalandars",   False, "Batsman",     "Right-hand bat", "",                       None,  None),
    ("Shahab Khan",        "Lahore Qalandars",   False, "Bowler",      "Right-hand bat", "Right-arm fast-medium",  None,  None),
    # Quetta Gladiators
    ("Alzarri Joseph",     "Quetta Gladiators",  True,  "Bowler",      "Right-hand bat", "Right-arm fast",         None,  8.5),
    ("Saqib Khan",         "Quetta Gladiators",  False, "Bowler",      "Right-hand bat", "Right-arm fast-medium",  None,  None),
    # Multan Sultans
    ("Shehzad Gul",        "Multan Sultans",     False, "Bowler",      "Right-hand bat", "Right-arm fast-medium",  None,  None),
    ("Imran Randhawa",     "Multan Sultans",     False, "Bowler",      "Left-hand bat",  "Slow left-arm orthodox", None,  None),
    ("Muhammad Shahzad",   "Multan Sultans",     False, "Batsman",     "Right-hand bat", "",                       None,  None),
    ("Muhammad Ismail",    "Multan Sultans",     False, "Bowler",      "Right-hand bat", "Right-arm fast",         None,  None),
    ("Atizaz Habib Khan",  "Multan Sultans",     False, "Bowler",      "Right-hand bat", "Right-arm fast-medium",  None,  None),
    # Rawalpindiz
    ("Jalat Khan",         "Rawalpindiz",        False, "Bowler",      "Right-hand bat", "Right-arm fast-medium",  None,  None),
    # Peshawar Zalmi
    ("Shoriful Islam",     "Peshawar Zalmi",     True,  "Bowler",      "Left-hand bat",  "Left-arm fast-medium",   None,  8.4),
    ("Shahnawaz Dahani",   "Peshawar Zalmi",     False, "Bowler",      "Right-hand bat", "Right-arm fast",         None,  None),
    ("Farhan Yousuf",      "Peshawar Zalmi",     False, "Batsman",     "Right-hand bat", "",                       None,  None),
    # Hyderabad Kingsmen
    ("Glenn Maxwell",      "Hyderabad Kingsmen", True,  "All-rounder", "Right-hand bat", "Right-arm off-break",    154.0, 7.9),
    ("Ahmed Hussain",      "Hyderabad Kingsmen", False, "Batsman",     "Right-hand bat", "",                       None,  None),
    # Islamabad United
    ("Nisar Ahmed",        "Islamabad United",   False, "Bowler",      "Right-hand bat", "Right-arm medium",       None,  None),
    ("Mohammad Faiq",      "Islamabad United",   False, "Batsman",     "Right-hand bat", "",                       None,  None),
]

# ── STEP 1: Update existing rows ──────────────────────────────────────────────
updated = 0
not_found_in_csv = []
for csv_name, team, overseas in SQUADS:
    if set_player(df, csv_name, team, overseas):
        updated += 1
    else:
        not_found_in_csv.append((csv_name, team))

# ── STEP 2: Deactivate disambiguation / stale entries ─────────────────────────
DISAMBIG = ["Mohammad Rizwan (MS)", "Ihsanullah (MS)", "Usman Khan (MS)", "Khushdil Shah (MS)"]
for d in DISAMBIG:
    mask = df["player_name"] == d
    if mask.sum():
        df.loc[mask, "is_active"] = "False"

# ── STEP 3: Deactivate everyone not in any 2026 squad ────────────────────────
active_csv_names = {csv_name for csv_name, _, _ in SQUADS} | {r[0] for r in NEW_ROWS}
deactivated = 0
for idx, row in df.iterrows():
    name = row["player_name"]
    if name in DISAMBIG:
        continue
    if name not in active_csv_names and str(row.get("is_active", "True")) != "False":
        df.loc[idx, "is_active"] = "False"
        deactivated += 1

# ── STEP 4: Add new rows ──────────────────────────────────────────────────────
cols = list(df.columns)
added = 0
updated_existing_new = []

for (name, team, overseas, role, bat, bowl, sr, eco) in NEW_ROWS:
    if (df["player_name"] == name).any():
        set_player(df, name, team, overseas)
        updated_existing_new.append(name)
        continue
    new_row = {c: "" for c in cols}
    new_row["player_name"]        = name
    new_row["current_team_2026"]  = team
    new_row["batting_style"]      = bat
    new_row["bowling_style"]      = bowl
    new_row["primary_role"]       = role
    new_row["nationality"]        = "Overseas" if overseas else "Pakistan"
    new_row["is_overseas"]        = str(overseas)
    new_row["is_active"]          = "True"
    new_row["psl_seasons_played"] = "0"
    new_row["data_tier"]          = "3"
    new_row["t20_career_sr"]      = str(sr) if sr is not None else ""
    new_row["t20_career_economy"] = str(eco) if eco is not None else ""
    # Emerging: True for non-overseas first-year Pakistani players (U23 proxy)
    new_row["is_emerging"]        = str(not overseas)
    import pandas as _pd
    df = _pd.concat([df, _pd.DataFrame([new_row])], ignore_index=True)
    added += 1

df.to_csv(CSV_PATH, index=False)

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("SQUAD UPDATE SUMMARY")
print("=" * 52)
print(f"Players updated (team/overseas change) : {updated}")
print(f"Players added (new rows)               : {added}")
print(f"Players deactivated                    : {deactivated}")
if updated_existing_new:
    print(f"New-list entries already in CSV (updated): {updated_existing_new}")
if not_found_in_csv:
    print(f"\nSquad entries NOT found in CSV ({len(not_found_in_csv)}) — check manually:")
    for n, t in not_found_in_csv:
        print(f"  - {n!r}  ({t})")

print(f"\nNew franchises added to theme: Rawalpindiz, Hyderabad Kingsmen")
print()

df2 = pd.read_csv(CSV_PATH)
active = df2[df2["is_active"].astype(str) == "True"]
print("Team headcounts (active players):")
tc = active.groupby("current_team_2026")["player_name"].count().sort_values(ascending=False)
for t, n in tc.items():
    print(f"  {t:<32} {n}")
print(f"\nTotal active : {len(active)}")
print(f"Total rows   : {len(df2)}")
