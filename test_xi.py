import sys
from datetime import datetime
from pathlib import Path

sys.path.append(r"c:\Users\Muhammad Wahib\Desktop\PSL DASHBOARD\psl_decision_engine")

from engine.decision_engine import generate_prematch_brief
from utils.situation import WeatherImpact

squad = [
    "Fakhar Zaman", "Abdullah Shafique", "Sikandar Raza",
    "Shaheen Shah Afridi", "Mustafizur Rahman", "Rashid Khan",
    "Haris Rauf", "Zaman Khan", "David Wiese",
    "Agha Salman", "Mohammad Nawaz", "Sahibzada Farhan",
    "Shamyl Hussain", "Ubaid Shah"
]

weather = WeatherImpact.neutral()

brief = generate_prematch_brief(
    our_team="Lahore Qalandars",
    opposition="Karachi Kings",
    venue="Gaddafi Stadium, Lahore",
    match_datetime=datetime.now(),
    our_squad=squad,
    weather_impact=weather
)

# Print Primary XI (Option A) constraints to see if EXACTLY 4 overseas and 1 emerging are applied
print("PRIMARY XI (Option A):")
for p in brief.xi_options[0].players:
    os = "[OS]" if p.is_overseas else ""
    em = "[E]" if p.is_emerging else ""
    print(f"{p.batting_position}. {p.player_name} {os} {em} ({p.role})")
print(f"Notes: {brief.xi_options[0].constraint_note}")
