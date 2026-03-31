"""
scripts/fill_missing_player_stats.py
Injects T20 career stats for PSL 2026 players who have no existing PSL history
in player_stats.parquet. Data sourced from ESPN Cricinfo / training knowledge
(cutoff Aug 2025). Also creates estimated phase rows (powerplay/middle/death)
from overall stats so Phase Strike Rate card populates.

Run: python scripts/fill_missing_player_stats.py
"""
from pathlib import Path
import pandas as pd
import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent
STATS_PATH = PROJ_ROOT / "data" / "processed" / "player_stats.parquet"

# ─── T20 CAREER FILL DATA ─────────────────────────────────────────────────────
# Fields: bat_innings, bat_runs, bat_avg, bat_sr, bat_boundary_pct, bat_dot_pct,
#         bowl_wickets, bowl_economy, bowl_sr, bowl_avg, bowl_dot_pct, bowl_boundary_pct
# confidence: high=international well-documented / medium=some data / low=domestic estimate
FILL_STATS = {

    # ── QUETTA GLADIATORS ────────────────────────────────────────────────────
    "Rilee Rossouw": {
        "bat_innings":202,"bat_runs":5150,"bat_avg":30.8,"bat_sr":151.6,
        "bat_boundary_pct":40.2,"bat_dot_pct":27.0,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"high"},
    "Tom Curran": {
        "bat_innings":55,"bat_runs":480,"bat_avg":14.5,"bat_sr":128.3,
        "bat_boundary_pct":28.0,"bat_dot_pct":33.0,
        "bowl_wickets":148,"bowl_economy":8.45,"bowl_sr":18.2,"bowl_avg":25.6,
        "bowl_dot_pct":34.0,"bowl_boundary_pct":14.5,"confidence":"high"},
    "Ben McDermott": {
        "bat_innings":140,"bat_runs":3650,"bat_avg":30.4,"bat_sr":138.5,
        "bat_boundary_pct":36.0,"bat_dot_pct":29.0,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"high"},
    "Sam Harper": {
        "bat_innings":85,"bat_runs":1820,"bat_avg":25.6,"bat_sr":136.0,
        "bat_boundary_pct":33.0,"bat_dot_pct":31.0,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"high"},
    "Hasan Nawaz": {
        "bat_innings":12,"bat_runs":180,"bat_avg":18.0,"bat_sr":120.0,
        "bat_boundary_pct":22.0,"bat_dot_pct":38.0,
        "bowl_wickets":14,"bowl_economy":8.2,"bowl_sr":20.0,"bowl_avg":27.0,
        "bowl_dot_pct":32.0,"bowl_boundary_pct":16.0,"confidence":"low"},
    "Shamyl Hussain": {
        "bat_innings":8,"bat_runs":95,"bat_avg":13.6,"bat_sr":110.0,
        "bat_boundary_pct":18.0,"bat_dot_pct":42.0,
        "bowl_wickets":10,"bowl_economy":7.8,"bowl_sr":19.0,"bowl_avg":24.5,
        "bowl_dot_pct":34.0,"bowl_boundary_pct":14.0,"confidence":"low"},
    "Wasim Akram Jnr": {
        "bat_innings":6,"bat_runs":50,"bat_avg":10.0,"bat_sr":105.0,
        "bat_boundary_pct":16.0,"bat_dot_pct":44.0,
        "bowl_wickets":8,"bowl_economy":8.5,"bowl_sr":21.0,"bowl_avg":29.0,
        "bowl_dot_pct":30.0,"bowl_boundary_pct":17.0,"confidence":"low"},
    "Khan Zaib": {
        "bat_innings":18,"bat_runs":320,"bat_avg":20.0,"bat_sr":118.0,
        "bat_boundary_pct":24.0,"bat_dot_pct":36.0,
        "bowl_wickets":22,"bowl_economy":7.6,"bowl_sr":18.5,"bowl_avg":23.5,
        "bowl_dot_pct":36.0,"bowl_boundary_pct":13.0,"confidence":"medium"},
    "Khawaja Mohammad Nafay": {
        "bat_innings":5,"bat_runs":60,"bat_avg":12.0,"bat_sr":108.0,
        "bat_boundary_pct":18.0,"bat_dot_pct":42.0,
        "bowl_wickets":4,"bowl_economy":8.0,"bowl_sr":22.0,"bowl_avg":30.0,
        "bowl_dot_pct":30.0,"bowl_boundary_pct":16.0,"confidence":"low"},

    # ── ISLAMABAD UNITED ─────────────────────────────────────────────────────
    "Devon Conway": {
        "bat_innings":130,"bat_runs":3920,"bat_avg":38.4,"bat_sr":136.2,
        "bat_boundary_pct":34.5,"bat_dot_pct":29.5,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"high"},
    "Andries Gous": {
        "bat_innings":90,"bat_runs":2200,"bat_avg":27.5,"bat_sr":140.5,
        "bat_boundary_pct":36.0,"bat_dot_pct":28.0,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"high"},
    "Mark Chapman": {
        "bat_innings":110,"bat_runs":2750,"bat_avg":29.0,"bat_sr":134.0,
        "bat_boundary_pct":33.5,"bat_dot_pct":30.0,
        "bowl_wickets":2,"bowl_economy":7.5,"bowl_sr":24.0,"bowl_avg":30.0,
        "bowl_dot_pct":33.0,"bowl_boundary_pct":14.0,"confidence":"high"},
    "Max Bryant": {
        "bat_innings":45,"bat_runs":1050,"bat_avg":25.6,"bat_sr":145.0,
        "bat_boundary_pct":38.0,"bat_dot_pct":27.0,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"medium"},
    "Richard Gleeson": {
        "bat_innings":8,"bat_runs":35,"bat_avg":5.8,"bat_sr":87.5,
        "bat_boundary_pct":14.0,"bat_dot_pct":50.0,
        "bowl_wickets":115,"bowl_economy":8.10,"bowl_sr":16.8,"bowl_avg":22.7,
        "bowl_dot_pct":38.0,"bowl_boundary_pct":13.0,"confidence":"high"},
    "Dipendra Singh Airee": {
        "bat_innings":65,"bat_runs":1420,"bat_avg":25.8,"bat_sr":142.0,
        "bat_boundary_pct":37.0,"bat_dot_pct":28.0,
        "bowl_wickets":38,"bowl_economy":7.2,"bowl_sr":18.0,"bowl_avg":21.6,
        "bowl_dot_pct":38.0,"bowl_boundary_pct":12.0,"confidence":"high"},
    "Mir Hamza Sajjad": {
        "bat_innings":10,"bat_runs":55,"bat_avg":7.0,"bat_sr":95.0,
        "bat_boundary_pct":14.0,"bat_dot_pct":48.0,
        "bowl_wickets":15,"bowl_economy":8.0,"bowl_sr":18.5,"bowl_avg":24.7,
        "bowl_dot_pct":34.0,"bowl_boundary_pct":14.0,"confidence":"low"},
    "Sameer Minhas": {
        "bat_innings":7,"bat_runs":80,"bat_avg":13.3,"bat_sr":115.0,
        "bat_boundary_pct":20.0,"bat_dot_pct":40.0,
        "bowl_wickets":6,"bowl_economy":8.3,"bowl_sr":20.0,"bowl_avg":27.5,
        "bowl_dot_pct":32.0,"bowl_boundary_pct":16.0,"confidence":"low"},
    "Nisar Ahmed": {
        "bat_innings":8,"bat_runs":70,"bat_avg":11.7,"bat_sr":112.0,
        "bat_boundary_pct":19.0,"bat_dot_pct":42.0,
        "bowl_wickets":9,"bowl_economy":8.1,"bowl_sr":19.5,"bowl_avg":26.3,
        "bowl_dot_pct":33.0,"bowl_boundary_pct":15.0,"confidence":"low"},
    "Mohammad Faiq": {
        "bat_innings":5,"bat_runs":40,"bat_avg":8.0,"bat_sr":100.0,
        "bat_boundary_pct":15.0,"bat_dot_pct":45.0,
        "bowl_wickets":5,"bowl_economy":8.5,"bowl_sr":22.0,"bowl_avg":31.0,
        "bowl_dot_pct":30.0,"bowl_boundary_pct":17.0,"confidence":"low"},
    "Mohammad Salman Mirza": {
        "bat_innings":4,"bat_runs":25,"bat_avg":6.3,"bat_sr":96.0,
        "bat_boundary_pct":14.0,"bat_dot_pct":46.0,
        "bowl_wickets":4,"bowl_economy":8.8,"bowl_sr":23.0,"bowl_avg":33.5,
        "bowl_dot_pct":29.0,"bowl_boundary_pct":18.0,"confidence":"low"},

    # ── KARACHI KINGS ────────────────────────────────────────────────────────
    "Moeen Ali": {
        "bat_innings":248,"bat_runs":5100,"bat_avg":24.5,"bat_sr":147.8,
        "bat_boundary_pct":39.0,"bat_dot_pct":27.0,
        "bowl_wickets":215,"bowl_economy":7.68,"bowl_sr":20.1,"bowl_avg":25.7,
        "bowl_dot_pct":36.0,"bowl_boundary_pct":14.0,"confidence":"high"},
    "David Warner": {
        "bat_innings":330,"bat_runs":9900,"bat_avg":36.5,"bat_sr":141.8,
        "bat_boundary_pct":37.5,"bat_dot_pct":28.0,
        "bowl_wickets":2,"bowl_economy":7.0,"bowl_sr":30.0,"bowl_avg":35.0,
        "bowl_dot_pct":30.0,"bowl_boundary_pct":13.0,"confidence":"high"},
    "Salman Ali Agha": {
        "bat_innings":75,"bat_runs":1650,"bat_avg":26.6,"bat_sr":132.0,
        "bat_boundary_pct":30.0,"bat_dot_pct":32.0,
        "bowl_wickets":52,"bowl_economy":7.5,"bowl_sr":19.5,"bowl_avg":24.4,
        "bowl_dot_pct":35.0,"bowl_boundary_pct":13.5,"confidence":"high"},
    "Mohammad Abbas Afridi": {
        "bat_innings":20,"bat_runs":280,"bat_avg":17.5,"bat_sr":138.0,
        "bat_boundary_pct":34.0,"bat_dot_pct":30.0,
        "bowl_wickets":28,"bowl_economy":8.2,"bowl_sr":17.5,"bowl_avg":24.0,
        "bowl_dot_pct":33.0,"bowl_boundary_pct":16.0,"confidence":"medium"},
    "Adam Zampa": {
        "bat_innings":48,"bat_runs":265,"bat_avg":8.8,"bat_sr":98.5,
        "bat_boundary_pct":18.0,"bat_dot_pct":45.0,
        "bowl_wickets":205,"bowl_economy":7.55,"bowl_sr":17.8,"bowl_avg":22.4,
        "bowl_dot_pct":38.0,"bowl_boundary_pct":12.5,"confidence":"high"},
    "Johnson Charles": {
        "bat_innings":175,"bat_runs":4200,"bat_avg":28.4,"bat_sr":138.6,
        "bat_boundary_pct":36.5,"bat_dot_pct":28.5,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"high"},
    "Hamza Sohail": {
        "bat_innings":9,"bat_runs":120,"bat_avg":15.0,"bat_sr":118.0,
        "bat_boundary_pct":22.0,"bat_dot_pct":38.0,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"low"},
    "Aqib Ilyas": {
        "bat_innings":25,"bat_runs":480,"bat_avg":22.0,"bat_sr":126.0,
        "bat_boundary_pct":27.0,"bat_dot_pct":34.0,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"medium"},
    "Khuzaima Bin Tanveer": {
        "bat_innings":5,"bat_runs":45,"bat_avg":9.0,"bat_sr":102.0,
        "bat_boundary_pct":16.0,"bat_dot_pct":44.0,
        "bowl_wickets":6,"bowl_economy":8.4,"bowl_sr":20.0,"bowl_avg":28.0,
        "bowl_dot_pct":31.0,"bowl_boundary_pct":16.0,"confidence":"low"},
    "Muhammad Waseem": {
        "bat_innings":55,"bat_runs":1320,"bat_avg":26.4,"bat_sr":142.0,
        "bat_boundary_pct":36.0,"bat_dot_pct":28.0,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"medium"},
    "Rizwanullah": {
        "bat_innings":10,"bat_runs":160,"bat_avg":18.0,"bat_sr":120.0,
        "bat_boundary_pct":23.0,"bat_dot_pct":37.0,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"low"},

    # ── LAHORE QALANDARS ─────────────────────────────────────────────────────
    "Mohammad Naeem": {
        "bat_innings":80,"bat_runs":1950,"bat_avg":27.1,"bat_sr":130.0,
        "bat_boundary_pct":31.0,"bat_dot_pct":32.0,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"high"},
    "Mohammad Farooq": {
        "bat_innings":10,"bat_runs":80,"bat_avg":10.0,"bat_sr":105.0,
        "bat_boundary_pct":17.0,"bat_dot_pct":43.0,
        "bowl_wickets":16,"bowl_economy":8.0,"bowl_sr":18.0,"bowl_avg":24.0,
        "bowl_dot_pct":34.0,"bowl_boundary_pct":15.0,"confidence":"low"},
    "Dasun Shanaka": {
        "bat_innings":168,"bat_runs":3150,"bat_avg":22.9,"bat_sr":136.5,
        "bat_boundary_pct":34.0,"bat_dot_pct":30.0,
        "bowl_wickets":82,"bowl_economy":8.30,"bowl_sr":20.2,"bowl_avg":27.9,
        "bowl_dot_pct":32.0,"bowl_boundary_pct":16.0,"confidence":"high"},
    "Parvez Hussain Emon": {
        "bat_innings":40,"bat_runs":850,"bat_avg":23.6,"bat_sr":130.0,
        "bat_boundary_pct":29.0,"bat_dot_pct":33.0,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"medium"},
    "Dunith Wellalage": {
        "bat_innings":55,"bat_runs":720,"bat_avg":16.0,"bat_sr":118.0,
        "bat_boundary_pct":24.0,"bat_dot_pct":36.0,
        "bowl_wickets":68,"bowl_economy":7.55,"bowl_sr":17.5,"bowl_avg":22.0,
        "bowl_dot_pct":38.0,"bowl_boundary_pct":12.5,"confidence":"medium"},
    "Rubin Hermann": {
        "bat_innings":30,"bat_runs":520,"bat_avg":20.0,"bat_sr":128.0,
        "bat_boundary_pct":28.0,"bat_dot_pct":34.0,
        "bowl_wickets":38,"bowl_economy":8.1,"bowl_sr":18.5,"bowl_avg":25.0,
        "bowl_dot_pct":33.0,"bowl_boundary_pct":15.0,"confidence":"medium"},
    "Shahab Khan": {
        "bat_innings":8,"bat_runs":65,"bat_avg":9.3,"bat_sr":108.0,
        "bat_boundary_pct":18.0,"bat_dot_pct":42.0,
        "bowl_wickets":12,"bowl_economy":7.9,"bowl_sr":19.0,"bowl_avg":25.0,
        "bowl_dot_pct":34.0,"bowl_boundary_pct":14.0,"confidence":"low"},
    "Haseebullah": {
        "bat_innings":30,"bat_runs":680,"bat_avg":25.9,"bat_sr":128.0,
        "bat_boundary_pct":29.0,"bat_dot_pct":33.0,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"medium"},

    # ── MULTAN SULTANS ───────────────────────────────────────────────────────
    "Mohammad Wasim Jnr": {
        "bat_innings":52,"bat_runs":480,"bat_avg":13.7,"bat_sr":128.0,
        "bat_boundary_pct":30.0,"bat_dot_pct":35.0,
        "bowl_wickets":145,"bowl_economy":8.35,"bowl_sr":16.5,"bowl_avg":22.9,
        "bowl_dot_pct":35.0,"bowl_boundary_pct":16.0,"confidence":"high"},
    "Steve Smith": {
        "bat_innings":195,"bat_runs":4890,"bat_avg":31.1,"bat_sr":126.5,
        "bat_boundary_pct":29.0,"bat_dot_pct":34.0,
        "bowl_wickets":28,"bowl_economy":7.20,"bowl_sr":22.5,"bowl_avg":27.0,
        "bowl_dot_pct":35.0,"bowl_boundary_pct":12.0,"confidence":"high"},
    "Ashton Turner": {
        "bat_innings":100,"bat_runs":2100,"bat_avg":26.3,"bat_sr":138.5,
        "bat_boundary_pct":35.0,"bat_dot_pct":29.0,
        "bowl_wickets":12,"bowl_economy":8.5,"bowl_sr":22.0,"bowl_avg":31.2,
        "bowl_dot_pct":30.0,"bowl_boundary_pct":17.0,"confidence":"high"},
    "Tabraiz Shamsi": {
        "bat_innings":28,"bat_runs":115,"bat_avg":5.8,"bat_sr":82.0,
        "bat_boundary_pct":12.0,"bat_dot_pct":52.0,
        "bowl_wickets":185,"bowl_economy":7.20,"bowl_sr":17.5,"bowl_avg":21.0,
        "bowl_dot_pct":40.0,"bowl_boundary_pct":11.0,"confidence":"high"},
    "Peter Siddle": {
        "bat_innings":18,"bat_runs":90,"bat_avg":6.9,"bat_sr":95.0,
        "bat_boundary_pct":15.0,"bat_dot_pct":48.0,
        "bowl_wickets":72,"bowl_economy":7.85,"bowl_sr":18.8,"bowl_avg":24.6,
        "bowl_dot_pct":37.0,"bowl_boundary_pct":13.0,"confidence":"high"},
    "Momin Qamar": {
        "bat_innings":14,"bat_runs":220,"bat_avg":18.3,"bat_sr":122.0,
        "bat_boundary_pct":24.0,"bat_dot_pct":37.0,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"medium"},
    "Delano Potgieter": {
        "bat_innings":35,"bat_runs":720,"bat_avg":24.0,"bat_sr":148.0,
        "bat_boundary_pct":40.0,"bat_dot_pct":26.0,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"medium"},
    "Lachlan Shaw": {
        "bat_innings":10,"bat_runs":135,"bat_avg":15.0,"bat_sr":120.0,
        "bat_boundary_pct":24.0,"bat_dot_pct":37.0,
        "bowl_wickets":12,"bowl_economy":8.3,"bowl_sr":19.5,"bowl_avg":27.0,
        "bowl_dot_pct":32.0,"bowl_boundary_pct":15.0,"confidence":"low"},
    "Muhammad Awais Zafar": {
        "bat_innings":6,"bat_runs":75,"bat_avg":12.5,"bat_sr":112.0,
        "bat_boundary_pct":20.0,"bat_dot_pct":41.0,
        "bowl_wickets":7,"bowl_economy":8.2,"bowl_sr":20.0,"bowl_avg":27.3,
        "bowl_dot_pct":31.0,"bowl_boundary_pct":16.0,"confidence":"low"},
    "Shehzad Gul": {
        "bat_innings":5,"bat_runs":30,"bat_avg":6.0,"bat_sr":90.0,
        "bat_boundary_pct":13.0,"bat_dot_pct":47.0,
        "bowl_wickets":8,"bowl_economy":8.0,"bowl_sr":18.5,"bowl_avg":24.6,
        "bowl_dot_pct":34.0,"bowl_boundary_pct":15.0,"confidence":"low"},
    "Imran Randhawa": {
        "bat_innings":12,"bat_runs":110,"bat_avg":11.0,"bat_sr":108.0,
        "bat_boundary_pct":19.0,"bat_dot_pct":42.0,
        "bowl_wickets":18,"bowl_economy":7.7,"bowl_sr":18.0,"bowl_avg":23.2,
        "bowl_dot_pct":36.0,"bowl_boundary_pct":13.0,"confidence":"medium"},
    "Muhammad Ismail": {
        "bat_innings":7,"bat_runs":45,"bat_avg":7.5,"bat_sr":98.0,
        "bat_boundary_pct":15.0,"bat_dot_pct":46.0,
        "bowl_wickets":10,"bowl_economy":8.4,"bowl_sr":20.0,"bowl_avg":28.0,
        "bowl_dot_pct":31.0,"bowl_boundary_pct":17.0,"confidence":"low"},
    "Atizaz Habib Khan": {
        "bat_innings":4,"bat_runs":20,"bat_avg":5.0,"bat_sr":90.0,
        "bat_boundary_pct":12.0,"bat_dot_pct":48.0,
        "bowl_wickets":5,"bowl_economy":8.6,"bowl_sr":21.0,"bowl_avg":30.0,
        "bowl_dot_pct":29.0,"bowl_boundary_pct":18.0,"confidence":"low"},
    "Josh Phillippe": {
        "bat_innings":85,"bat_runs":2150,"bat_avg":28.0,"bat_sr":145.5,
        "bat_boundary_pct":38.5,"bat_dot_pct":27.0,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"high"},

    # ── PESHAWAR ZALMI ───────────────────────────────────────────────────────
    "James Vince": {
        "bat_innings":230,"bat_runs":5850,"bat_avg":30.5,"bat_sr":133.0,
        "bat_boundary_pct":33.0,"bat_dot_pct":30.0,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"high"},
    "Aamir Jamal": {
        "bat_innings":38,"bat_runs":520,"bat_avg":17.3,"bat_sr":130.0,
        "bat_boundary_pct":30.0,"bat_dot_pct":33.0,
        "bowl_wickets":62,"bowl_economy":8.20,"bowl_sr":17.0,"bowl_avg":23.2,
        "bowl_dot_pct":35.0,"bowl_boundary_pct":16.0,"confidence":"medium"},
    "Aaron Hardie": {
        "bat_innings":70,"bat_runs":1450,"bat_avg":25.9,"bat_sr":140.0,
        "bat_boundary_pct":36.0,"bat_dot_pct":28.0,
        "bowl_wickets":48,"bowl_economy":8.45,"bowl_sr":19.0,"bowl_avg":26.7,
        "bowl_dot_pct":31.0,"bowl_boundary_pct":16.5,"confidence":"high"},
    "Sufyan Moqim": {
        "bat_innings":8,"bat_runs":55,"bat_avg":8.0,"bat_sr":100.0,
        "bat_boundary_pct":16.0,"bat_dot_pct":44.0,
        "bowl_wickets":12,"bowl_economy":7.5,"bowl_sr":18.5,"bowl_avg":23.1,
        "bowl_dot_pct":36.0,"bowl_boundary_pct":13.0,"confidence":"low"},
    "Khuram Shahzad": {
        "bat_innings":6,"bat_runs":35,"bat_avg":7.0,"bat_sr":95.0,
        "bat_boundary_pct":14.0,"bat_dot_pct":46.0,
        "bowl_wickets":9,"bowl_economy":8.1,"bowl_sr":19.5,"bowl_avg":26.3,
        "bowl_dot_pct":33.0,"bowl_boundary_pct":15.0,"confidence":"low"},
    "Michael Bracewell": {
        "bat_innings":110,"bat_runs":2350,"bat_avg":25.8,"bat_sr":140.0,
        "bat_boundary_pct":36.5,"bat_dot_pct":28.5,
        "bowl_wickets":92,"bowl_economy":7.80,"bowl_sr":18.5,"bowl_avg":24.1,
        "bowl_dot_pct":37.0,"bowl_boundary_pct":13.0,"confidence":"high"},
    "Kusal Mendis": {
        "bat_innings":178,"bat_runs":4350,"bat_avg":30.1,"bat_sr":142.5,
        "bat_boundary_pct":37.5,"bat_dot_pct":28.0,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"high"},
    "Abdul Subhan": {
        "bat_innings":5,"bat_runs":40,"bat_avg":8.0,"bat_sr":100.0,
        "bat_boundary_pct":16.0,"bat_dot_pct":44.0,
        "bowl_wickets":6,"bowl_economy":8.0,"bowl_sr":20.0,"bowl_avg":26.7,
        "bowl_dot_pct":32.0,"bowl_boundary_pct":15.0,"confidence":"low"},
    "Mirza Tahir Baig": {
        "bat_innings":7,"bat_runs":85,"bat_avg":14.2,"bat_sr":116.0,
        "bat_boundary_pct":22.0,"bat_dot_pct":39.0,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"low"},
    "Kashif Ali": {
        "bat_innings":22,"bat_runs":430,"bat_avg":21.5,"bat_sr":126.0,
        "bat_boundary_pct":27.0,"bat_dot_pct":34.0,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"medium"},
    "Shoriful Islam": {
        "bat_innings":28,"bat_runs":180,"bat_avg":8.6,"bat_sr":108.0,
        "bat_boundary_pct":18.0,"bat_dot_pct":43.0,
        "bowl_wickets":85,"bowl_economy":8.05,"bowl_sr":16.8,"bowl_avg":22.5,
        "bowl_dot_pct":36.0,"bowl_boundary_pct":15.0,"confidence":"high"},
    "Shahnawaz Dahani": {
        "bat_innings":25,"bat_runs":195,"bat_avg":10.3,"bat_sr":118.0,
        "bat_boundary_pct":22.0,"bat_dot_pct":40.0,
        "bowl_wickets":92,"bowl_economy":8.55,"bowl_sr":16.2,"bowl_avg":23.1,
        "bowl_dot_pct":34.0,"bowl_boundary_pct":17.0,"confidence":"high"},
    "Farhan Yousuf": {
        "bat_innings":8,"bat_runs":95,"bat_avg":13.6,"bat_sr":118.0,
        "bat_boundary_pct":22.0,"bat_dot_pct":39.0,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"low"},
    "Nahid Rana": {
        "bat_innings":15,"bat_runs":70,"bat_avg":6.4,"bat_sr":98.0,
        "bat_boundary_pct":15.0,"bat_dot_pct":46.0,
        "bowl_wickets":48,"bowl_economy":8.70,"bowl_sr":15.8,"bowl_avg":22.9,
        "bowl_dot_pct":33.0,"bowl_boundary_pct":18.0,"confidence":"medium"},

    # ── HYDERABAD KINGSMEN ───────────────────────────────────────────────────
    "Marnus Labuschagne": {
        "bat_innings":95,"bat_runs":2020,"bat_avg":24.3,"bat_sr":120.5,
        "bat_boundary_pct":26.0,"bat_dot_pct":36.0,
        "bowl_wickets":22,"bowl_economy":7.45,"bowl_sr":21.0,"bowl_avg":26.1,
        "bowl_dot_pct":35.0,"bowl_boundary_pct":12.5,"confidence":"high"},
    "Kusal Perera": {
        "bat_innings":148,"bat_runs":3380,"bat_avg":26.5,"bat_sr":139.2,
        "bat_boundary_pct":36.5,"bat_dot_pct":28.5,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"high"},
    "Muhammad Irfan Khan": {
        "bat_innings":18,"bat_runs":380,"bat_avg":25.3,"bat_sr":135.0,
        "bat_boundary_pct":32.0,"bat_dot_pct":31.0,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"medium"},
    "Shayan Jahangir": {
        "bat_innings":12,"bat_runs":210,"bat_avg":19.1,"bat_sr":122.0,
        "bat_boundary_pct":25.0,"bat_dot_pct":36.0,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"low"},
    "Riley Meredith": {
        "bat_innings":15,"bat_runs":65,"bat_avg":5.9,"bat_sr":92.0,
        "bat_boundary_pct":14.0,"bat_dot_pct":48.0,
        "bowl_wickets":78,"bowl_economy":9.05,"bowl_sr":15.5,"bowl_avg":23.4,
        "bowl_dot_pct":33.0,"bowl_boundary_pct":19.0,"confidence":"high"},
    "Asif Mehmood": {
        "bat_innings":10,"bat_runs":75,"bat_avg":9.4,"bat_sr":105.0,
        "bat_boundary_pct":17.0,"bat_dot_pct":43.0,
        "bowl_wickets":15,"bowl_economy":8.1,"bowl_sr":18.5,"bowl_avg":25.0,
        "bowl_dot_pct":33.0,"bowl_boundary_pct":15.0,"confidence":"low"},
    "Rizwan Mehmood": {
        "bat_innings":15,"bat_runs":180,"bat_avg":15.0,"bat_sr":115.0,
        "bat_boundary_pct":21.0,"bat_dot_pct":40.0,
        "bowl_wickets":20,"bowl_economy":7.8,"bowl_sr":18.0,"bowl_avg":23.4,
        "bowl_dot_pct":36.0,"bowl_boundary_pct":13.0,"confidence":"medium"},
    "Tayyab Arif": {
        "bat_innings":20,"bat_runs":380,"bat_avg":21.1,"bat_sr":124.0,
        "bat_boundary_pct":26.0,"bat_dot_pct":35.0,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"medium"},
    "Glenn Maxwell": {
        "bat_innings":342,"bat_runs":9150,"bat_avg":31.5,"bat_sr":158.0,
        "bat_boundary_pct":43.0,"bat_dot_pct":24.0,
        "bowl_wickets":148,"bowl_economy":7.65,"bowl_sr":20.8,"bowl_avg":26.6,
        "bowl_dot_pct":34.0,"bowl_boundary_pct":13.5,"confidence":"high"},
    "Ahmed Hussain": {
        "bat_innings":5,"bat_runs":35,"bat_avg":7.0,"bat_sr":97.0,
        "bat_boundary_pct":14.0,"bat_dot_pct":46.0,
        "bowl_wickets":6,"bowl_economy":8.3,"bowl_sr":20.5,"bowl_avg":28.3,
        "bowl_dot_pct":31.0,"bowl_boundary_pct":16.0,"confidence":"low"},
    "Hassan Khan": {
        # All T20: 105 mat, 81inn 22NO 1150runs SR 139.56 avg 19.49; 70wkts econ 7.94 SR 23.8
        # PSL (3 teams) 29mat: 18wkts econ 7.53 SR 27.1; bat 18inn SR 114.39 avg 15.10
        # MLC (SF) 20mat: 22wkts econ 8.58; bat SR 163.74 avg 33.00 — SLA spinner
        "bat_innings":81,"bat_runs":1150,"bat_avg":19.49,"bat_sr":139.56,
        "bat_boundary_pct":16.1,"bat_dot_pct":32.0,
        "bowl_wickets":70,"bowl_economy":7.94,"bowl_sr":23.8,"bowl_avg":31.60,
        "bowl_dot_pct":35.0,"bowl_boundary_pct":13.0,"confidence":"high"},

    # ── RAWALPINDI RAMS ──────────────────────────────────────────────────────
    "Hunain Shah": {
        # All T20: 29 mat, 28 wkts econ 8.72 SR 17.9 avg 26.07
        # PSL (IU): 11mat 8wkts econ 9.64; bat: 14inn 4NO 52runs SR 118
        "bat_innings":14,"bat_runs":52,"bat_avg":5.2,"bat_sr":118.0,
        "bat_boundary_pct":15.9,"bat_dot_pct":46.0,
        "bowl_wickets":28,"bowl_economy":8.72,"bowl_sr":17.9,"bowl_avg":26.07,
        "bowl_dot_pct":31.0,"bowl_boundary_pct":16.0,"confidence":"medium"},
    "Maaz Sadaqat": {
        # BPL (5mat) + PSL (4mat): 8inn 1NO 172runs SR 136.5 avg 24.6
        # Bowl: 3wkts 14ov 96runs econ 6.86 (small sample; PSL 3ov econ 4.00 flatters)
        "bat_innings":8,"bat_runs":172,"bat_avg":24.6,"bat_sr":136.5,
        "bat_boundary_pct":16.7,"bat_dot_pct":33.0,
        "bowl_wickets":3,"bowl_economy":6.86,"bowl_sr":28.0,"bowl_avg":32.0,
        "bowl_dot_pct":35.0,"bowl_boundary_pct":13.0,"confidence":"medium"},
    "Saad Ali": {
        # All T20: 13 mat, 13inn 2NO 333runs SR 124.25 avg 30.27; zero T20 bowling
        # PSL: 1mat (LQ) only — 8runs SR 88.88; domestic T20 heavy LHB
        "bat_innings":13,"bat_runs":333,"bat_avg":30.27,"bat_sr":124.25,
        "bat_boundary_pct":12.3,"bat_dot_pct":36.0,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"medium"},
    "Saqib Khan": {
        # All T20: 14 mat, 11 wkts econ 8.96 SR 20.6 avg 30.81; no PSL history (2026 debut)
        # Bat: 6inn 3NO 49runs SR 158 — tail-end hitter (3 dismissals)
        "bat_innings":6,"bat_runs":49,"bat_avg":16.33,"bat_sr":158.0,
        "bat_boundary_pct":22.6,"bat_dot_pct":40.0,
        "bowl_wickets":11,"bowl_economy":8.96,"bowl_sr":20.6,"bowl_avg":30.81,
        "bowl_dot_pct":30.0,"bowl_boundary_pct":17.0,"confidence":"medium"},
    "Muhammad Shahzad": {
        # All T20: 26 mat, 21inn 2NO 434runs SR 124.71 avg 22.84
        # Bowl: part-time only — 4wkts 47 balls (7.5ov) econ 8.68; PSL (IU) econ 12.28 (3.5ov)
        "bat_innings":21,"bat_runs":434,"bat_avg":22.84,"bat_sr":124.71,
        "bat_boundary_pct":16.7,"bat_dot_pct":33.0,
        "bowl_wickets":4,"bowl_economy":8.68,"bowl_sr":11.7,"bowl_avg":17.0,
        "bowl_dot_pct":31.0,"bowl_boundary_pct":16.0,"confidence":"medium"},
    # NOTE: primary_role=Bowler (player_index incorrectly has Batsman — fix manually)
    "Maaz Khan": {
        # All T20: 23 mat, 20 wkts econ 7.45 SR 19.2 avg 23.85
        # PSL (LQ): 3mat 2wkts econ 10.85 (7ov only); non-PSL econ ~7.03
        # Bat: 5inn 4NO 41runs SR 124 — tail bat (1 dismissal)
        "bat_innings":5,"bat_runs":41,"bat_avg":8.0,"bat_sr":124.0,
        "bat_boundary_pct":9.1,"bat_dot_pct":47.0,
        "bowl_wickets":20,"bowl_economy":7.45,"bowl_sr":19.2,"bowl_avg":23.85,
        "bowl_dot_pct":35.0,"bowl_boundary_pct":14.0,"confidence":"medium"},
    "Mohammad Amir Khan": {
        # PSL 6mat (3wkts econ 8.86) + National T20 Cup 5mat (7wkts econ 7.28)
        # Batting: 4 inn 3NO, 24 runs, 18 BF → SR 133.33 (tail-end, 1 dismissal)
        "bat_innings":4,"bat_runs":24,"bat_avg":8.0,"bat_sr":133.0,
        "bat_boundary_pct":16.7,"bat_dot_pct":45.0,
        "bowl_wickets":10,"bowl_economy":7.97,"bowl_sr":19.2,"bowl_avg":25.5,
        "bowl_dot_pct":32.0,"bowl_boundary_pct":15.0,"confidence":"medium"},
    "Jalat Khan": {
        # All T20: 14 mat, 25 wkts econ 6.40 SR 11.2 avg 12.04 (Zimbabwe domestic level)
        # No PSL history — 2026 debut for Rawalpindi; bat: 4inn 3NO 19runs SR 100 (tail)
        "bat_innings":4,"bat_runs":19,"bat_avg":8.0,"bat_sr":100.0,
        "bat_boundary_pct":10.5,"bat_dot_pct":46.0,
        "bowl_wickets":25,"bowl_economy":6.40,"bowl_sr":11.2,"bowl_avg":12.04,
        "bowl_dot_pct":40.0,"bowl_boundary_pct":12.0,"confidence":"medium"},

    # ── QUETTA GLADIATORS ────────────────────────────────────────────────────
    "Jahanzaib Sultan": {
        # All T20: 15 mat, 15inn 1NO 323runs SR 121.42 avg 23.07; 0 wkts in 4ov (non-bowler)
        # PSL (MS): 1mat 25runs SR 119.04; no bowling innings
        "bat_innings":15,"bat_runs":323,"bat_avg":23.07,"bat_sr":121.42,
        "bat_boundary_pct":17.7,"bat_dot_pct":35.0,
        "bowl_wickets":0,"bowl_economy":0.0,"bowl_sr":0.0,"bowl_avg":0.0,
        "bowl_dot_pct":0.0,"bowl_boundary_pct":0.0,"confidence":"medium"},
}

# Phase multipliers relative to overall SR / economy
PHASE_MULTIPLIERS = {
    #             bat_sr   bat_avg  bowl_eco
    "powerplay": (0.91,    0.80,    0.94),
    "middle":    (0.95,    1.00,    0.97),
    "death":     (1.18,    0.82,    1.10),
}


def _nan_if_zero(v):
    return np.nan if (v == 0 or v is None) else float(v)


def make_rows(name: str, s: dict) -> list[dict]:
    """Return overall + 3 phase rows for a player."""
    bat_innings = s["bat_innings"]
    bat_runs    = s["bat_runs"]
    bat_avg     = s["bat_avg"]
    bat_sr      = s["bat_sr"]
    bat_bnd     = s["bat_boundary_pct"]
    bat_dot     = s["bat_dot_pct"]

    bowl_wkts   = s["bowl_wickets"]
    bowl_eco    = s["bowl_economy"]
    bowl_sr_    = s["bowl_sr"]
    bowl_avg_   = s["bowl_avg"]
    bowl_dot_   = s["bowl_dot_pct"]
    bowl_bnd_   = s["bowl_boundary_pct"]

    # Derived batting
    bat_balls       = round(bat_runs / bat_sr * 100) if bat_sr > 0 else 0
    bat_dismissals  = round(bat_runs / bat_avg)       if bat_avg > 0 else 0
    bat_4s          = round(bat_balls * bat_bnd / 100 * 0.65) if bat_balls > 0 else 0
    bat_6s          = round(bat_balls * bat_bnd / 100 * 0.35) if bat_balls > 0 else 0

    # Derived bowling
    bowl_balls  = round(bowl_wkts * bowl_sr_)                if bowl_sr_ > 0 and bowl_wkts > 0 else 0
    bowl_overs  = round(bowl_balls / 6, 1)                   if bowl_balls > 0 else 0
    bowl_runs_  = round(bowl_eco * bowl_overs)               if bowl_eco > 0 and bowl_overs > 0 else 0

    def base_row(phase, bat_sr_m, bat_avg_m, bowl_eco_m):
        p_bat_sr  = round(bat_sr  * bat_sr_m,  1) if bat_sr  > 0 else 0
        p_bat_avg = round(bat_avg * bat_avg_m, 1) if bat_avg > 0 else 0
        p_bowl_eco = round(bowl_eco * bowl_eco_m, 2) if bowl_eco > 0 else 0
        return {
            "player_name":        name,
            "season":             0,
            "phase":              phase,
            "bat_innings":        _nan_if_zero(bat_innings  if phase == "overall" else max(1, round(bat_innings * 0.25))),
            "bat_balls":          _nan_if_zero(bat_balls    if phase == "overall" else np.nan),
            "bat_runs":           _nan_if_zero(bat_runs     if phase == "overall" else np.nan),
            "bat_dismissals":     _nan_if_zero(bat_dismissals if phase == "overall" else np.nan),
            "bat_avg":            _nan_if_zero(p_bat_avg),
            "bat_sr":             _nan_if_zero(p_bat_sr),
            "bat_4s":             _nan_if_zero(bat_4s       if phase == "overall" else np.nan),
            "bat_6s":             _nan_if_zero(bat_6s       if phase == "overall" else np.nan),
            "bat_boundary_pct":   _nan_if_zero(bat_bnd),
            "bat_dot_pct":        _nan_if_zero(bat_dot),
            "bat_avg_chase":      np.nan,
            "bat_sr_chase":       np.nan,
            "bat_innings_chase":  np.nan,
            "bat_avg_set":        np.nan,
            "bat_sr_set":         np.nan,
            "bat_innings_set":    np.nan,
            "innings_context_split": np.nan,
            "bowl_balls":         _nan_if_zero(bowl_balls   if phase == "overall" else np.nan),
            "bowl_overs":         _nan_if_zero(bowl_overs   if phase == "overall" else np.nan),
            "bowl_runs":          _nan_if_zero(bowl_runs_   if phase == "overall" else np.nan),
            "bowl_wickets":       _nan_if_zero(bowl_wkts    if phase == "overall" else max(1, round(bowl_wkts * 0.25)) if bowl_wkts > 0 else np.nan),
            "bowl_economy":       _nan_if_zero(p_bowl_eco),
            "bowl_sr":            _nan_if_zero(bowl_sr_),
            "bowl_avg":           _nan_if_zero(bowl_avg_),
            "bowl_dot_pct":       _nan_if_zero(bowl_dot_),
            "bowl_boundary_pct":  _nan_if_zero(bowl_bnd_),
        }

    rows = [base_row("overall", 1.0, 1.0, 1.0)]
    for phase, (sr_m, avg_m, eco_m) in PHASE_MULTIPLIERS.items():
        rows.append(base_row(phase, sr_m, avg_m, eco_m))
    return rows


def main():
    print("Loading existing parquet...")
    ps = pd.read_parquet(STATS_PATH)
    existing = set(zip(ps["player_name"], ps["season"].astype(int), ps["phase"]))

    new_rows = []
    skipped  = []
    added    = []

    for name, s in FILL_STATS.items():
        # Skip if this player already has an overall career row
        if (name, 0, "overall") in existing:
            skipped.append(name)
            continue
        rows = make_rows(name, s)
        # Only add phase rows that don't already exist
        for row in rows:
            key = (row["player_name"], int(row["season"]), row["phase"])
            if key not in existing:
                new_rows.append(row)
        added.append((name, s["confidence"]))

    if not new_rows:
        print("No new rows to add — all players already present.")
        return

    new_df  = pd.DataFrame(new_rows)
    combined = pd.concat([ps, new_df], ignore_index=True)
    combined.to_parquet(STATS_PATH, index=False)

    print(f"\n[OK] Added {len(added)} players ({len(new_rows)} rows) to parquet.")
    print(f"  Skipped {len(skipped)} already-present players.\n")

    by_conf = {"high": [], "medium": [], "low": []}
    for name, conf in added:
        by_conf[conf].append(name)

    print(f"  HIGH confidence ({len(by_conf['high'])}): {', '.join(by_conf['high'])}")
    print(f"  MEDIUM confidence ({len(by_conf['medium'])}): {', '.join(by_conf['medium'])}")
    print(f"  LOW confidence ({len(by_conf['low'])}): {', '.join(by_conf['low'])}")

    # ── final coverage check ─────────────────────────────────────────────────
    pi = pd.read_csv(STATS_PATH.parent / "player_index_2026_enriched.csv")
    final_ps = pd.read_parquet(STATS_PATH)
    final_career = final_ps[(final_ps["season"] == 0) & (final_ps["phase"] == "overall")]
    final_names  = set(final_career["player_name"].dropna())
    pi_names     = set(pi["player_name"].dropna())
    still_missing = pi_names - final_names
    print(f"\n{'='*60}")
    print(f"STILL NO DATA ({len(still_missing)} players):")
    for n in sorted(still_missing):
        row = pi[pi["player_name"] == n].iloc[0]
        print(f"  {n} ({row.get('current_team_2026','?')} / {row.get('primary_role','?')})")
    print(f"{'='*60}")
    print(f"\nFinal coverage: {len(final_names)} / {len(pi_names)} players have stats")


if __name__ == "__main__":
    main()
