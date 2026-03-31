import csv
import os
import pandas as pd

CSV_PATH = r"c:\Users\Muhammad Wahib\Desktop\PSL DASHBOARD\psl_decision_engine\data\processed\player_index_2026_enriched.csv"

# Raw user input strings for the 8 squads
KARACHI_KINGS = "David Warner [C] ✈, Hasan Ali, Mohammad Abbas Afridi, Khushdil Shah, Saad Baig, Moeen Ali ✈, Azam Khan, Salman Ali Agha, Shahid Aziz, Mir Hamza, Adam Zampa ✈, Mohammad Hamza Sohail, Aqib Ilyas, Khuzaima Bin Tanveer ✈, Johnson Charles ✈, Muhammad Waseem ✈, Ihsanullah, Rizwanullah"
LAHORE_QALANDARS = "Shaheen Shah Afridi [C], Abdullah Shafique, Sikandar Raza ✈, Mohammad Naeem, Mustafizur Rahman ✈, Haris Rauf, Usama Mir, Fakhar Zaman, Ubaid Shah, Haseebullah, Mohammad Farooq, Dasun Shanaka ✈, Parvez Hussain Emon ✈, Asif Ali, Hussain Talat, Tayyab Tahir, Dunith Wellalage ✈, Rubin Hermann ✈, Maaz Khan, Shahab Khan"
QUETTA_GLADIATORS = "Saud Shakeel [C], Usman Tariq, Hasan Nawaz, Shamyl Hussain, Alzarri Joseph ✈, Rilee Rossouw ✈, Ahmed Daniyal, Jahanzaib Sultan, Jahandad Khan, Khawaja Mohammad Nafay, Wasim Akram Jnr, Khan Zeb, Bismillah Khan, Saqib Khan, Brett Hampton ✈, Sam Harper ✈, Bevon Jacobs ✈, Abrar Ahmed, Ben McDermott ✈, Tom Curran ✈"
MULTAN_SULTANS = "Ashton Turner [C] ✈, Mohammad Nawaz, Shehzad Gul, Faisal Akram, Imran Randhawa, Arafat Minhas, Sahibzada Farhan, Steve Smith ✈, Peter Siddle ✈, Tabraiz Shamsi ✈, Lachlan Shaw ✈, Delano Potgieter ✈, Josh Phillippe ✈, Shan Masood, Momin Qamar, Muhammad Awais Zafar, Muhammad Shahzad, Arshad Iqbal, Mohammad Wasim Jnr, Muhammad Ismail, Atizaz Habib Khan"
RAWALPINDIZ = "Mohammad Rizwan [C], Sam Billings ✈, Jalat Khan, Yasir Khan, Naseem Shah, Rishad Hossain ✈, Daryl Mitchell ✈, Mohammad Amir, Abdullah Fazal, Amad Butt, Dian Forrestor ✈, Laurie Evans ✈, Asif Afridi, Kamran Ghulam, Fawad Ali, Mohammad Amir Khan, Shahzaib Khan, Jake Fraser-McGurk ✈, Saad Masood"
PESHAWAR_ZALMI = "Babar Azam [C], Sufyan Muqeem, Abdul Samad, Ali Raza, Aaron Hardie ✈, Aamir Jamal, Khurram Shahzad, Mohammad Haris, Khalid Usman, Abdul Subhan, James Vince ✈, Michael Bracewell ✈, Kusal Mendis ✈, Iftikhar Ahmed, Nahid Rana ✈, Mirza Tahir Baig, Kashif Ali, Shahnawaz Dahani, Farhan Yousuf, Shoriful Islam ✈"
HYDERABAD_KINGSMEN = "Marnus Labuschagne [C] ✈, Usman Khan, Akif Javed, Maaz Sadaqat, Saim Ayub, Mohammad Ali, Kusal Perera ✈, Muhammad Irfan Khan, Hassan Khan ✈, Shayan Jahangir ✈, Glenn Maxwell ✈, Hammad Azam, Riley Meredith ✈, Sharjeel Khan, Asif Mehmood, Hunain Shah, Rizwan Mehmood, Saad Ali, Tayyab Arif, Ahmed Hussain"
ISLAMABAD_UNITED = "Shadab Khan [C], Salman Irshad, Andries Gous ✈, Devon Conway ✈, Faheem Ashraf, Mehran Mumtaz, Max Bryant ✈, Mark Chapman ✈, Nisar Ahmed, Mir Hamza Sajjad, Sameen Gul, Sameer Minhas, Imad Wasim, Richard Gleeson ✈, Haider Ali, Mohammad Hasnain, Dipendra Singh Airee ✈, Mohammad Faiq, Mohammad Salman Mirza"

# Dictionary of team -> raw string
RAW_SQUADS = {
    "Karachi Kings": KARACHI_KINGS,
    "Lahore Qalandars": LAHORE_QALANDARS,
    "Quetta Gladiators": QUETTA_GLADIATORS,
    "Multan Sultans": MULTAN_SULTANS,
    "Rawalpindiz": RAWALPINDIZ,
    "Peshawar Zalmi": PESHAWAR_ZALMI,
    "Hyderabad Kingsmen": HYDERABAD_KINGSMEN,
    "Islamabad United": ISLAMABAD_UNITED,
}

# New overseas players
NEW_OVERSEAS = {
    "Marnus Labuschagne": {"primary_role": "Batsman", "batting_style": "Right-hand bat", "bowling_style": "Right-arm leg-break", "t20_career_sr": "118.0", "t20_career_economy": "7.8"},
    "Glenn Maxwell": {"primary_role": "All-rounder", "batting_style": "Right-hand bat", "bowling_style": "Right-arm off-break", "t20_career_sr": "154.0", "t20_career_economy": "7.9"},
    "Riley Meredith": {"primary_role": "Bowler", "batting_style": "Right-hand bat", "bowling_style": "Right-arm fast", "t20_career_economy": "8.8"},
    "Jake Fraser-McGurk": {"primary_role": "Batsman", "batting_style": "Right-hand bat", "t20_career_sr": "182.0"},
    "Aaron Hardie": {"primary_role": "All-rounder", "batting_style": "Right-hand bat", "bowling_style": "Right-arm medium", "t20_career_sr": "138.0", "t20_career_economy": "8.2"},
    "Steve Smith": {"primary_role": "Batsman", "batting_style": "Right-hand bat", "bowling_style": "Right-arm leg-break", "t20_career_sr": "125.0", "t20_career_economy": "8.5"},
    "Devon Conway": {"primary_role": "Batsman", "batting_style": "Left-hand bat", "bowling_style": "Right-arm medium", "t20_career_sr": "136.0"},
    "Mark Chapman": {"primary_role": "Batsman", "batting_style": "Left-hand bat", "t20_career_sr": "129.0"},
    "Shoriful Islam": {"primary_role": "Bowler", "batting_style": "Left-hand bat", "bowling_style": "Left-arm fast-medium", "t20_career_economy": "8.4"},
    "Nahid Rana": {"primary_role": "Bowler", "batting_style": "Right-hand bat", "bowling_style": "Right-arm fast", "t20_career_economy": "9.1"},
    "Tom Curran": {"primary_role": "Bowler", "batting_style": "Right-hand bat", "bowling_style": "Right-arm fast-medium", "t20_career_economy": "8.6"},
    "Ben McDermott": {"primary_role": "WK-Batsman", "batting_style": "Right-hand bat", "t20_career_sr": "141.0"},
    "Sam Harper": {"primary_role": "WK-Batsman", "batting_style": "Right-hand bat", "t20_career_sr": "138.0"},
    "Lachlan Shaw": {"primary_role": "Batsman", "batting_style": "Left-hand bat", "t20_career_sr": "142.0"},
    "Delano Potgieter": {"primary_role": "All-rounder", "batting_style": "Right-hand bat", "bowling_style": "Right-arm off-break", "t20_career_sr": "155.0", "t20_career_economy": "8.1"},
    "Josh Phillippe": {"primary_role": "WK-Batsman", "batting_style": "Right-hand bat", "t20_career_sr": "148.0"},
    "Peter Siddle": {"primary_role": "Bowler", "batting_style": "Right-hand bat", "bowling_style": "Right-arm fast-medium", "t20_career_economy": "8.3"},
    "Bevon Jacobs": {"primary_role": "Batsman", "batting_style": "Left-hand bat", "t20_career_sr": "132.0"},
    "Dipendra Singh Airee": {"primary_role": "All-rounder", "batting_style": "Right-hand bat", "bowling_style": "Right-arm off-break", "t20_career_sr": "145.0", "t20_career_economy": "7.6"},
    "Richard Gleeson": {"primary_role": "Bowler", "batting_style": "Right-hand bat", "bowling_style": "Right-arm fast", "t20_career_economy": "8.9"},
    "Rubin Hermann": {"primary_role": "Bowler", "batting_style": "Left-hand bat", "bowling_style": "Left-arm fast-medium", "t20_career_economy": "8.5"},
    "Dunith Wellalage": {"primary_role": "All-rounder", "batting_style": "Left-hand bat", "bowling_style": "Slow left-arm orthodox", "t20_career_sr": "118.0", "t20_career_economy": "7.2"},
    "Dian Forrestor": {"primary_role": "All-rounder", "batting_style": "Right-hand bat", "bowling_style": "Right-arm fast-medium", "t20_career_sr": "128.0", "t20_career_economy": "8.0"},
    "Laurie Evans": {"primary_role": "Batsman", "batting_style": "Right-hand bat", "t20_career_sr": "138.0"},
    "Rishad Hossain": {"primary_role": "Bowler", "batting_style": "Right-hand bat", "bowling_style": "Right-arm leg-break", "t20_career_economy": "7.4"},
    "Hassan Khan": {"primary_role": "Bowler", "batting_style": "Right-hand bat", "bowling_style": "Slow left-arm orthodox", "t20_career_economy": "7.8"},
    "Shayan Jahangir": {"primary_role": "Batsman", "batting_style": "Left-hand bat", "t20_career_sr": "122.0"},
    "Johnson Charles": {"primary_role": "Batsman", "batting_style": "Right-hand bat", "t20_career_sr": "143.0"},
    "Muhammad Waseem": {"primary_role": "Batsman", "batting_style": "Right-hand bat", "t20_career_sr": "148.0"},
    "Khuzaima Bin Tanveer": {"primary_role": "Bowler", "batting_style": "Right-hand bat", "bowling_style": "Right-arm fast-medium", "t20_career_economy": "8.7"},
    "Sam Billings": {"primary_role": "WK-Batsman", "batting_style": "Right-hand bat", "t20_career_sr": "145.0"},
    "Brett Hampton": {"primary_role": "All-rounder", "batting_style": "Right-hand bat", "bowling_style": "Right-arm fast-medium", "t20_career_sr": "130.0", "t20_career_economy": "8.4"},
}

ALIASES = {
    "Sikander Raza": "Sikandar Raza",
    "AS Joseph": "Alzarri Joseph",
    "RR Rossouw": "Rilee Rossouw",
    "DJ Mitchell": "Daryl Mitchell",
    "Khurram Shahzad": "Khuram Shahzad",
    "Mohammad Hamza Sohail": "Hamza Sohail",
    "Sufyan Muqeem": "Sufyan Moqim",
    "Khan Zeb": "Khan Zaib"
}

def clean_name(n):
    n = n.replace("[C]", "").replace("✈", "").strip()
    return ALIASES.get(n, n)

official_roster = {}
for team, squad_str in RAW_SQUADS.items():
    players = [p.strip() for p in squad_str.split(",")]
    for p in players:
        is_overseas = "✈" in p
        name = clean_name(p)
        official_roster[name] = {"team": team, "is_overseas": is_overseas}

print(f"Parsed {len(official_roster)} players from official squads.")

df = pd.read_csv(CSV_PATH)
df['player_name'] = df['player_name'].str.strip()
print(f"Loaded {len(df)} players from CSV.")

# 1. De-activate players not in official roster
# (Or just drop them from the dataframe as user said "delete that players profiles from the csv")
df = df[df['player_name'].isin(official_roster.keys())].copy()
print(f"After dropping unused, {len(df)} players remain.")

existing_names = set(df['player_name'].tolist())

# 2. Update existing players
for idx, row in df.iterrows():
    name = row['player_name']
    roster_info = official_roster[name]
    df.at[idx, 'current_team_2026'] = roster_info['team']
    df.at[idx, 'is_overseas'] = True if roster_info['is_overseas'] else False
    df.at[idx, 'is_active'] = True

# 3. Add missing players
missing_players = [p for p in official_roster.keys() if p not in existing_names]
print(f"Adding {len(missing_players)} new players.")

new_rows = []
for p in missing_players:
    info = official_roster[p]
    row = {
        'player_name': p,
        'current_team_2026': info['team'],
        'is_active': True,
        'is_overseas': True if info['is_overseas'] else False,
        'data_tier': 3,
    }
    
    # Fill stats if it's a known new overseas
    if p in NEW_OVERSEAS:
        stats = NEW_OVERSEAS[p]
        for k, v in stats.items():
            row[k] = v
            
    if 'primary_role' not in row:
        row['primary_role'] = 'Batsman' # fallback
    if 'batting_style' not in row:
        row['batting_style'] = 'Right-hand bat' # fallback

    # Add empty fields for other columns to prevent NaN issues
    for col in df.columns:
        if col not in row:
            row[col] = ""
            
    new_rows.append(row)

if new_rows:
    df_new = pd.DataFrame(new_rows)
    df = pd.concat([df, df_new], ignore_index=True)

# 4. Save
df.to_csv(CSV_PATH, index=False)

print("\nSQUAD UPDATE SUMMARY")
print("====================")
print(f"Total players in updated CSV: {len(df)}")
print(f"Active players: {len(df[df['is_active'] == True])}")
print("New franchises ready: Rawalpindiz, Hyderabad Kingsmen")

grouped = df.groupby('current_team_2026')['player_name'].count()
print("\nPlayers per team:")
print(grouped)
