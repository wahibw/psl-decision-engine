# app.py
# Dash entry point — two-mode navigation shell
# MODE 1: /prep  — Match Prep Room (pre-match brief generator)
# MODE 2: /dugout — Dugout Screen (live match intelligence)
# PAGE 3: /players — Player Profiles (accessible from both modes)

import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from flask import Response, request as _flask_request
import json as _json
import pandas as _pd
import numpy as _np
from pathlib import Path as _Path

# Load .env file first so all API keys are available to every module
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass

# Claude Sonnet 4.6 — T20 specialist advisor (gracefully no-ops if key absent)
from utils.claude_advisor import (
    get_dugout_advice as _claude_dugout,
    get_prematch_coaching_read as _claude_prematch,
    get_on_demand_advice as _claude_on_demand,
    get_coach_evaluation as _claude_eval,
    get_full_coaching_analysis as _claude_full,
)

from utils.theme import (
    BRAND_ORANGE, BRAND_ACCENT, DARK_BG, DARK_ALT,
    TEXT_PRIMARY, TEXT_SECONDARY, BORDER_COLOR,
)

# ---------------------------------------------------------------------------
# APP INIT
# ---------------------------------------------------------------------------

_GOOGLE_FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=Orbitron:wght@400;500;600;700;800;900"
    "&family=Rajdhani:wght@300;400;500;600;700"
    "&family=Exo+2:wght@300;400;500;600;700;800"
    "&display=swap"
)

app = dash.Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.DARKLY, _GOOGLE_FONTS],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
    title="PSL Decision Intelligence",
)

server = app.server   # for gunicorn: web: gunicorn app:server

# ---------------------------------------------------------------------------
# STARTUP WARMUP — runs in background thread immediately after server boots.
# Loads the TabNet/XGBoost model and fetches weather for the default venue
# so the first "Generate Brief" click is fast (model already in memory,
# weather already cached).  Any failure here is silent — the brief endpoint
# will still work, just slightly slower on the very first call.
# ---------------------------------------------------------------------------

def _warmup():
    import threading as _thr, time as _wt
    def _do_warmup():
        try:
            # 1. Pre-load TabNet / XGBoost model into the module-level cache
            from models.train_xi_scorer import load_model as _lm
            from pathlib import Path as _P
            _mp = str(_P(__file__).resolve().parent / "models" / "saved" / "xi_scorer.pkl")
            _payload = _lm(_mp)
            import engine.xi_selector as _xis
            _xis._CACHED_MODEL_PAYLOAD = _payload
            _xis._CACHED_MODEL_KEY = _mp
            print("[warmup] XI scorer model loaded into cache.")
        except Exception as _e:
            print(f"[warmup] model warmup skipped: {_e}")
        try:
            # 2. Pre-fetch weather for the most common PSL venues
            from weather.weather_impact import get_match_weather_impact as _gwi
            from datetime import datetime as _dtt
            _venues = [
                "Gaddafi Stadium, Lahore",
                "National Stadium, Karachi",
                "Rawalpindi Cricket Stadium",
                "Multan Cricket Stadium",
            ]
            _dt = _dtt.now().replace(hour=19, minute=0, second=0, microsecond=0)
            for _v in _venues:
                try:
                    _gwi(_v, _dt)
                except Exception:
                    pass
            print("[warmup] Weather cache primed for default venues.")
        except Exception as _e:
            print(f"[warmup] weather warmup skipped: {_e}")
    _t = _thr.Thread(target=_do_warmup, daemon=True)
    _t.start()

_warmup()

# ---------------------------------------------------------------------------
# PLAYERS DATA API  — served at /api/players-data
# The standalone HTML players page fetches this endpoint for real stats.
# ---------------------------------------------------------------------------

_PROJ_ROOT = _Path(__file__).resolve().parent
_players_cache = None   # lazy-loaded once per process restart


def _build_players_payload() -> str:
    """Read parquets/CSVs and build the full JSON payload for the players page."""

    ps = _pd.read_parquet(_PROJ_ROOT / "data" / "processed" / "player_stats.parquet")
    pi = _pd.read_csv(_PROJ_ROOT / "data" / "processed" / "player_index_2026_enriched_v2.csv")

    # Verified hand-curated career stats (from CricInfo screenshots) used as
    # fallback for players with no cricsheet data in the parquet.
    _verified_path = _PROJ_ROOT / "data" / "processed" / "player_stats_verified.csv"
    _verified = _pd.read_csv(_verified_path)[
        ["player_name", "bat_innings", "bat_runs", "bat_avg", "bat_sr",
         "bat_boundary_pct", "bat_dot_pct",
         "bowl_wickets", "bowl_economy", "bowl_sr", "bowl_avg"]
    ].rename(columns=lambda c: c if c == "player_name" else f"v_{c}")

    career = ps[(ps["season"] == 0) & (ps["phase"] == "overall")].copy()
    phases = ps[(ps["season"] == 0) & (ps["phase"] != "overall")].copy()

    df = _pd.merge(pi, career, on="player_name", how="left", suffixes=("_csv", "_prq"))
    df = _pd.merge(df, _verified, on="player_name", how="left")

    ROLE_COLORS = {
        "Batsman":      "#3B82F6",
        "WK-Batsman":   "#8B5CF6",
        "Wicketkeeper": "#8B5CF6",
        "All-Rounder":  "#FFD700",
        "All-rounder":  "#FFD700",
        "Bowler":       "#F44336",
    }
    ROLE_POS = {
        "Batsman": "BAT", "WK-Batsman": "WK", "Wicketkeeper": "WK",
        "All-Rounder": "AR", "All-rounder": "AR", "Bowler": "BWL",
    }

    def sf(v, d=0.0):
        try:
            f = float(v)
            return d if (_pd.isna(f) or _np.isinf(f)) else f
        except Exception:
            return d

    def si(v, d=0):
        try:
            return int(float(v)) if not _pd.isna(float(v)) else d
        except Exception:
            return d

    def zones_from_stats(row):
        sr  = sf(row.get("bat_sr"), 120)
        bnd = sf(row.get("bat_boundary_pct"), 25)
        sty = str(row.get("batting_style") or "Right")
        left = "Left" in sty

        # Base distribution per hand (FL,SL,MW,LO,LF,CV,PT,TM)
        base = [8, 10, 15, 18, 22, 12, 7, 8] if left else [7, 12, 22, 18, 15, 10, 8, 8]
        agg  = min(1.5, max(0.5, sr / 130))
        bf   = min(1.4, max(0.6, bnd / 25))

        z = base[:]
        z[2] = int(base[2] * agg * 0.9 + base[2] * bf * 0.1)
        z[3] = int(base[3] * agg * 0.85 + base[3] * 0.15)
        z[4] = int(base[4] * agg * 0.85 + base[4] * 0.15)

        total = sum(z) or 100
        z = [max(1, round(v * 100 / total)) for v in z]
        z[0] += 100 - sum(z)   # fix rounding to ensure sum == 100
        return dict(zip(["fl", "sl", "mw", "lo", "lf", "cv", "pt", "tm"], z))

    def bat_rating(row):
        sr  = sf(row.get("bat_sr"))
        avg = sf(row.get("bat_avg"))
        bnd = sf(row.get("bat_boundary_pct"))
        if sr == 0 and avg == 0:
            return 0
        return min(99, max(1, int(sr / 2.2 * 0.5 + min(50, avg) * 1.0 + bnd * 0.3)))

    def bowl_rating(row):
        eco  = sf(row.get("bowl_economy"))
        wkts = sf(row.get("bowl_wickets"))
        if eco == 0 and wkts == 0:
            return 0
        return min(99, max(1, int((10 - eco) * 12 + wkts * 0.5)))

    def bowl_heatmap(wkts, bowling_style):
        """Compute a 6x3 wicket heatmap from career wickets + bowling style.
        Rows: Short / Short-Good / Good Length / Full / Yorker / Full Toss
        Cols: Off / Middle / Leg
        """
        if wkts <= 0:
            return [[0,0,0]]*6
        bsty = str(bowling_style or "").lower()
        is_spin = any(x in bsty for x in ["spin", "break", "wrist"])
        is_fast = any(x in bsty for x in ["fast", "medium"])

        if is_spin:
            # Spinners: cluster at good length off/middle, some full
            weights = [
                [0.05, 0.10, 0.06],  # Short
                [0.12, 0.28, 0.14],  # Short-Good
                [0.30, 0.55, 0.32],  # Good Length
                [0.18, 0.30, 0.20],  # Full
                [0.05, 0.10, 0.06],  # Yorker
                [0.02, 0.04, 0.03],  # Full Toss
            ]
        elif is_fast:
            # Fast/medium: good length + yorker off/middle, short-good
            weights = [
                [0.08, 0.18, 0.10],  # Short
                [0.18, 0.42, 0.20],  # Short-Good
                [0.32, 0.72, 0.35],  # Good Length
                [0.10, 0.22, 0.12],  # Full
                [0.14, 0.28, 0.15],  # Yorker
                [0.02, 0.04, 0.02],  # Full Toss
            ]
        else:
            # Generic medium/unknown
            weights = [
                [0.06, 0.14, 0.08],
                [0.14, 0.35, 0.16],
                [0.28, 0.60, 0.30],
                [0.14, 0.28, 0.15],
                [0.10, 0.20, 0.12],
                [0.02, 0.04, 0.02],
            ]

        total_w = sum(v for row in weights for v in row)
        result = []
        for row in weights:
            result.append([max(0, round(v / total_w * wkts)) for v in row])

        # Clamp to ensure sum ≈ wkts (distribute rounding error on peak cell)
        actual = sum(v for row in result for v in row)
        diff = wkts - actual
        if diff != 0:
            result[2][1] = max(0, result[2][1] + diff)
        return result

    def phase_sr(player):
        d = {}
        for ph in ("powerplay", "middle", "death"):
            r = phases[(phases["player_name"] == player) & (phases["phase"] == ph)]
            d[ph] = round(sf(r.iloc[0].get("bat_sr") if not r.empty else 0), 1)
        return d

    # Season-level overall rows for T20 trend (last 3 PSL seasons)
    T20_SEASONS = [2023, 2024, 2025]
    seasonal = ps[(ps["season"].isin(T20_SEASONS)) & (ps["phase"] == "overall")].copy()

    def t20_seasons(player):
        out = {}
        for yr in T20_SEASONS:
            r = seasonal[(seasonal["player_name"] == player) & (seasonal["season"] == yr)]
            if r.empty:
                out[yr] = None
            else:
                row_s = r.iloc[0]
                _bowl_sr_val = sf(row_s.get("bowl_sr"))
                out[yr] = {
                    "avg":        round(sf(row_s.get("bat_avg")), 1),
                    "sr":         round(sf(row_s.get("bat_sr")), 1),
                    "wkts":       si(row_s.get("bowl_wickets")),
                    "econ":       round(sf(row_s.get("bowl_economy")), 2),
                    "bnd":        round(sf(row_s.get("bat_boundary_pct")), 1),
                    "dot":        round(sf(row_s.get("bat_dot_pct")), 1),
                    "bowl_sr":    round(_bowl_sr_val, 1),
                    "bowl_avg":   round(sf(row_s.get("bowl_economy")) * _bowl_sr_val / 6, 1) if _bowl_sr_val > 0 else 0,
                    "bowl_dot":   round(sf(row_s.get("bowl_dot_pct")), 1),
                    "bowl_bnd":   round(sf(row_s.get("bowl_boundary_pct")), 1),
                }
        return out

    players = []
    for _, row in df.iterrows():
        name = str(row.get("player_name") or "")
        if not name:
            continue
        role      = str(row.get("primary_role") or "Batsman")
        team      = str(row.get("current_team_2026") or "PSL")
        init      = "".join(w[0].upper() for w in name.split()[:2])
        _tw       = team.split()
        _tfolder  = (_tw[0] + "-" + _tw[1].lower()) if len(_tw) >= 2 else team.lower().replace(" ", "-")
        _fname    = name.lower().replace(" ", "-") + ".png"
        photo     = f"assets/players/{_tfolder}/{_fname}"
        ph        = phase_sr(name)
        t20       = t20_seasons(name)
        bstyle    = str(row.get("bowling_style") or "")
        wkts_val  = si(row.get("bowl_wickets"))

        # Fallback chains for players with no cricsheet data:
        # parquet career stats → verified hand-curated CSV → ESPN league career → ESPN T20 career
        _bat_avg  = sf(row.get("bat_avg"))  or sf(row.get("v_bat_avg"))  or sf(row.get("psl_career_avg"))  or sf(row.get("t20_career_avg"))
        _bat_sr   = sf(row.get("bat_sr"))   or sf(row.get("v_bat_sr"))   or sf(row.get("psl_career_sr"))   or sf(row.get("t20_career_sr"))
        _bat_bnd  = sf(row.get("bat_boundary_pct")) or sf(row.get("v_bat_boundary_pct"))
        _bat_dot  = sf(row.get("bat_dot_pct"))      or sf(row.get("v_bat_dot_pct"))
        _bowl_eco = sf(row.get("bowl_economy")) or sf(row.get("v_bowl_economy")) or sf(row.get("psl_career_economy")) or sf(row.get("t20_career_economy"))
        _bowl_sr  = sf(row.get("bowl_sr"))  or sf(row.get("v_bowl_sr"))
        _bowl_wkt = si(row.get("bowl_wickets")) or si(row.get("v_bowl_wickets"))
        _bowl_avg = round(_bowl_eco * _bowl_sr / 6, 1) if _bowl_sr > 0 else round(sf(row.get("v_bowl_avg")), 1)
        _bat_runs = si(row.get("bat_runs"))   or si(row.get("v_bat_runs"))
        _bat_inn  = si(row.get("bat_innings")) or si(row.get("v_bat_innings"))

        players.append({
            "name":         name,
            "role":         role,
            "pos":          ROLE_POS.get(role, "BAT"),
            "team":         team,
            "initials":     init,
            "color":        ROLE_COLORS.get(role, "#3B82F6"),
            "bat_runs":     _bat_runs,
            "bat_innings":  _bat_inn,
            "bat_avg":      round(_bat_avg, 1),
            "bat_sr":       round(_bat_sr, 1),
            "bat_bnd":      round(_bat_bnd, 1),
            "bat_dot":      round(_bat_dot, 1),
            "bowl_wickets": _bowl_wkt,
            "bowl_economy": round(_bowl_eco, 2),
            "bowl_sr":      round(_bowl_sr, 1),
            "bowl_dot":     round(sf(row.get("bowl_dot_pct_prq")) or sf(row.get("bowl_dot_pct_csv")) or sf(row.get("bowl_dot_pct")), 1),
            "bowl_bnd":     round(sf(row.get("bowl_boundary_pct")), 1),
            "bowl_avg":     _bowl_avg,
            "bat_rating":   bat_rating(row),
            "bowl_rating":  bowl_rating(row),
            "pp_sr":        ph["powerplay"],
            "mid_sr":       ph["middle"],
            "death_sr":     ph["death"],
            "zones":        zones_from_stats(row),
            "t20":          t20,
            "bowl_heatmap": bowl_heatmap(wkts_val, bstyle),
            "ipl_matches":        si(row.get("ipl_matches")),
            "ipl_career_avg":     round(sf(row.get("ipl_career_avg")), 1),
            "ipl_career_sr":      round(sf(row.get("ipl_career_sr")), 1),
            "ipl_career_economy": round(sf(row.get("ipl_career_economy")), 2),
            "bpl_matches":        si(row.get("bpl_matches")),
            "bpl_career_avg":     round(sf(row.get("bpl_career_avg")), 1),
            "bpl_career_sr":      round(sf(row.get("bpl_career_sr")), 1),
            "bpl_career_economy": round(sf(row.get("bpl_career_economy")), 2),
            "lpl_matches":        si(row.get("lpl_matches")),
            "lpl_career_avg":     round(sf(row.get("lpl_career_avg")), 1),
            "lpl_career_sr":      round(sf(row.get("lpl_career_sr")), 1),
            "lpl_career_economy": round(sf(row.get("lpl_career_economy")), 2),
            "psl_matches":        si(row.get("psl_matches")),
            "psl_career_avg":     round(sf(row.get("psl_career_avg")), 1),
            "psl_career_sr":      round(sf(row.get("psl_career_sr")), 1),
            "psl_career_economy": round(sf(row.get("psl_career_economy")), 2),
            "t20_career_avg":     round(sf(row.get("t20_career_avg")), 1),
            "t20_career_sr":      round(sf(row.get("t20_career_sr")), 1),
            "t20_career_economy": round(sf(row.get("t20_career_economy")), 2),
            "photo":        photo,
        })

    players.sort(key=lambda p: (p["team"], p["name"]))
    return _json.dumps({"players": players})


_PSL_VENUES = [
    "Gaddafi Stadium, Lahore",
    "National Stadium, Karachi",
    "Rawalpindi Cricket Stadium",
    "Multan Cricket Stadium",
    "Arbab Niaz Stadium, Peshawar",
    "Iqbal Stadium, Faisalabad",
    "Dubai International Cricket Stadium",
    "Sharjah Cricket Stadium",
    "Sheikh Zayed Stadium, Abu Dhabi",
]


@server.route("/api/prep-meta")
def _prep_meta_api():
    """Return teams, venues, and per-team squad lists for the prep room form."""
    try:
        pi = _pd.read_csv(_PROJ_ROOT / "data" / "processed" / "player_index_2026_enriched.csv")
        if "is_active" in pi.columns:
            pi = pi[pi["is_active"] == True].copy()
        squad_by_team: dict = {}
        for _, row in pi.iterrows():
            team = str(row.get("current_team_2026") or "").strip()
            name = str(row.get("player_name") or "").strip()
            if team and name:
                squad_by_team.setdefault(team, []).append(name)
        from utils.theme import PSL_2026_TEAMS
        return Response(
            _json.dumps({"teams": PSL_2026_TEAMS, "venues": _PSL_VENUES, "squad_by_team": squad_by_team}),
            mimetype="application/json",
        )
    except Exception as exc:
        return Response(
            _json.dumps({"error": str(exc), "teams": [], "venues": [], "squad_by_team": {}}),
            mimetype="application/json", status=500,
        )


@server.route("/api/prep-brief", methods=["POST"])
def _prep_brief_api():
    """Generate a full pre-match brief from the posted match setup."""
    import traceback as _tb
    try:
        data = _json.loads(_flask_request.get_data(as_text=True))
        our_team   = str(data.get("our_team", "")).strip()
        opposition = str(data.get("opposition", "")).strip()
        venue      = str(data.get("venue", "")).strip()
        squad      = [str(p).strip() for p in data.get("squad", []) if str(p).strip()]
        dt_str     = str(data.get("match_datetime", "")).strip()

        from datetime import datetime as _dt
        try:
            dt = _dt.fromisoformat(dt_str)
        except Exception:
            dt = _dt.now().replace(hour=19, minute=0, second=0, microsecond=0)

        if len(squad) < 11:
            return Response(
                _json.dumps({"error": f"Need at least 11 squad players, got {len(squad)}"}),
                mimetype="application/json", status=400,
            )

        # Fetch weather and load engine in parallel — weather is the slowest
        # external call (~3s HTTP) and can overlap with module imports.
        import concurrent.futures as _cf
        from utils.situation import WeatherImpact

        def _fetch_weather():
            try:
                from weather.weather_impact import get_match_weather_impact
                return get_match_weather_impact(venue, dt)
            except Exception:
                return WeatherImpact.neutral()

        with _cf.ThreadPoolExecutor(max_workers=1) as _pool:
            _weather_fut = _pool.submit(_fetch_weather)
            from engine.decision_engine import generate_prematch_brief   # import while weather fetches
            weather = _weather_fut.result()   # block only if not yet done

        # Resolve captain — always locked into every XI option
        from utils.theme import PSL_2026_CAPTAINS
        captain = PSL_2026_CAPTAINS.get(our_team)
        # Ensure captain is actually in the submitted squad (guard against stale map)
        if captain and captain not in squad:
            captain = None
        forced = [captain] if captain else []

        brief = generate_prematch_brief(
            our_team       = our_team,
            opposition     = opposition,
            venue          = venue,
            match_datetime = dt,
            our_squad      = squad,
            weather_impact = weather,
            season         = 0,
            innings        = 1,
            forced_players = forced,
            captain        = captain,
        )

        serialised = _serialize_brief(brief)

        # ── Claude Sonnet 4.6 pre-match coaching brief ────────────────────
        try:
            xi_primary = brief.xi_options[0] if brief.xi_options else None
            xi_summary = (
                ", ".join(p.name for p in xi_primary.players[:11])
                if xi_primary else "XI not generated"
            )
            bp = brief.bowling_plan
            bowling_summary = (
                f"Openers: {bp.overs[0].primary_bowler}/{bp.overs[1].primary_bowler}; "
                f"Death: {bp.overs[-1].primary_bowler}/{bp.overs[-2].primary_bowler}"
            ) if bp and len(bp.overs) >= 20 else "Bowling plan unavailable"

            matchup_texts = [mn.note for mn in brief.matchup_notes[:4]] if brief.matchup_notes else []
            opp_threats   = [
                f"{b.name} (pos {b.batting_position}, {b.danger_level} danger)"
                for b in (brief.opposition_order.predicted_order[:4]
                          if brief.opposition_order else [])
            ]
            weather_sum = (
                f"Dew onset over {brief.weather_impact.dew_onset_over} "
                f"(spinner penalty {brief.weather_impact.spinner_penalty:.0%}), "
                f"swing ×{brief.weather_impact.swing_bonus:.2f}"
                if brief.weather_impact else "Conditions normal"
            )
            scenarios_sum = ""
            if brief.batting_scenarios:
                card_a = next((c for c in brief.batting_scenarios if c.scenario_id == "A"), None)
                if card_a:
                    scenarios_sum = card_a.key_message[:120]

            coaching_read = _claude_prematch(
                our_team               = our_team,
                opposition             = opposition,
                venue                  = venue,
                toss_recommendation    = brief.toss.recommendation if brief.toss else "",
                toss_reasoning         = brief.toss.reasoning[:120] if brief.toss else "",
                xi_summary             = xi_summary,
                bowling_plan_summary   = bowling_summary,
                key_matchups           = matchup_texts,
                weather_summary        = weather_sum,
                opposition_top_threats = opp_threats,
                batting_scenarios_summary = scenarios_sum,
            )
            if coaching_read:
                serialised["coaching_read"] = coaching_read
        except Exception:
            pass   # coaching_read is optional — never break the brief

        return Response(_json.dumps(serialised), mimetype="application/json")

    except Exception as exc:
        return Response(
            _json.dumps({"error": str(exc), "trace": _tb.format_exc()}),
            mimetype="application/json", status=500,
        )


def _compute_ball_probs(over_plan: list, player_stats_path: "_Path") -> list:
    """
    Build 120 per-ball boundary-risk probabilities from real bowler PSL stats.

    For each over the assigned bowler's phase economy is fetched from
    player_stats.parquet (PSL career aggregate, season 0).  Economy is
    normalised to a 0–1 risk value with a phase-specific multiplier so that
    death-over assignments read higher than power-play ones even at equal
    economy rates.  Six balls per over share a base probability with ±5%
    ball-level noise.

    Fallback chain:  parquet career → t20_career_economy column in parquet
    → 8.5 (league average placeholder).
    """
    import random as _rnd

    # Load PSL career stats once
    try:
        _ps = _pd.read_parquet(player_stats_path)
        # Keep only career rows (season 0)
        _career = _ps[_ps["season"] == 0].copy()
    except Exception:
        _career = _pd.DataFrame()

    _PHASE_MAP = {
        "PP":         "powerplay",
        "Mid":        "middle",
        "Pre-Death":  "middle",
        "Death":      "death",
    }
    # Phase multiplier: death is inherently riskier regardless of bowler quality
    _PHASE_MULT = {"PP": 0.95, "Mid": 0.82, "Pre-Death": 0.90, "Death": 1.30}
    # Economy → raw risk: normalise from [5.5, 13.0] → [0.0, 1.0]
    _ECO_LO, _ECO_HI = 5.5, 13.0

    probs = []
    for ov in sorted(over_plan, key=lambda x: x["over"]):
        bowler = ov.get("bowler", "") or ""
        phase  = ov.get("phase", "Mid")
        stat_phase = _PHASE_MAP.get(phase, "middle")
        mult   = _PHASE_MULT.get(phase, 0.90)

        eco = 8.5  # default league average
        if bowler and not _career.empty:
            row = _career[
                (_career["player_name"] == bowler) &
                (_career["phase"] == stat_phase)
            ]
            if not row.empty and "bowl_economy" in row.columns:
                v = row["bowl_economy"].iloc[0]
                if _pd.notna(v) and float(v) > 0:
                    eco = float(v)
            elif not row.empty and "bowl_overs" in row.columns:
                # Try the overall phase row if phase-specific missing
                overall = _career[
                    (_career["player_name"] == bowler) &
                    (_career["phase"] == "overall")
                ]
                if not overall.empty and "bowl_economy" in overall.columns:
                    v2 = overall["bowl_economy"].iloc[0]
                    if _pd.notna(v2) and float(v2) > 0:
                        eco = float(v2)

        raw  = (eco - _ECO_LO) / (_ECO_HI - _ECO_LO)
        base = max(0.08, min(0.92, raw * mult))

        for _ in range(6):
            noise = (_rnd.random() - 0.5) * 0.10
            probs.append(round(max(0.05, min(0.97, base + noise)), 3))

    # Pad to exactly 120 if fewer overs were returned
    while len(probs) < 120:
        probs.append(0.35)
    return probs[:120]


def _serialize_brief(brief) -> dict:
    """Convert PreMatchBrief dataclass tree to a plain JSON-serialisable dict."""
    w = brief.weather_impact

    xi_opts = []
    for opt in (brief.xi_options or []):
        xi_opts.append({
            "label":           opt.label,
            "description":     opt.description,
            "overseas_count":  opt.overseas_count,
            "bowler_count":    opt.bowler_count,
            "constraint_note": opt.constraint_note,
            "players": [
                {"pos": p.batting_position, "name": p.player_name,
                 "role": p.role, "key_stat": p.key_stat, "source": p.model_source}
                for p in opt.players
            ],
        })

    bp = brief.bowling_plan
    over_plan = [
        {"over": ov.over, "bowler": ov.primary_bowler, "backup": ov.backup_bowler,
         "phase": ov.phase, "note": ov.reason}
        for ov in (bp.overs if bp else [])
    ]

    opp = brief.opposition_order
    opp_order = [
        {"position": b.position, "name": b.player_name, "danger": b.danger_rating,
         "arrival": b.arrival_over_range, "style": b.batting_style,
         "career_sr": round(float(b.career_sr or 0), 1),
         "death_sr":  round(float(b.death_sr  or 0), 1),
         "note": b.key_note, "confidence": b.confidence}
        for b in (opp.predicted_order if opp else [])
    ]

    matchups = [
        {"batter": mn.batter, "bowler": mn.bowler, "advantage": mn.advantage,
         "conf": mn.confidence.lower(), "balls": mn.balls, "note": mn.note}
        for mn in (brief.matchup_notes or [])
    ]

    _SC_COLORS = {"A": "#4CAF50", "B": "#FFC107", "C": "#F44336", "D": "#00E5FF"}
    scenarios = []
    for sc in (brief.batting_scenarios or []):
        scenarios.append({
            "id":          sc.scenario_id,
            "title":       sc.name,
            "color":       _SC_COLORS.get(sc.scenario_id, "#FFD700"),
            "desc":        sc.description,
            "trigger":     sc.trigger,
            "key_message": sc.key_message,
            "weather_note":sc.weather_note,
            "players": [
                {"name": p.player_name, "pos": p.position,
                 "role": p.role_in_card, "note": p.instruction}
                for p in sc.batting_order
            ],
        })

    return {
        "toss": {
            "recommendation": brief.toss.recommendation,
            "reasoning":      brief.toss.reasoning,
            "dl_note":        brief.toss.dl_note,
        },
        "xi_options":     xi_opts,
        "over_plan":      over_plan,
        "bowler_summary": (bp.bowler_summary if bp else {}),
        "contingencies":  (bp.contingencies  if bp else []),
        "key_decisions":  (bp.key_decisions  if bp else []),
        "opposition":     opp_order,
        "matchup_notes":  matchups,
        "scenarios":      scenarios,
        "weather": {
            "humidity":        round(float(w.raw_humidity or 0), 1),
            "temp":            round(float(w.raw_temp    or 0), 1),
            "wind_kph":        round(float(w.raw_wind_kph or 0), 1),
            "wind_dir":        str(w.raw_wind_dir or ""),
            "dew_onset_over":  w.dew_onset_over,
            "spinner_penalty": round(w.spinner_penalty, 2),
            "swing_bonus":     round(w.swing_bonus,     2),
            "dl_needed":       w.dl_planning_needed,
            "warnings":        w.warnings,
        },
        "captain":         (brief.captain or ""),
        "data_tier_notes": (brief.data_tier_notes or []),
        "ball_probs": _compute_ball_probs(
            over_plan,
            _PROJ_ROOT / "data" / "processed" / "player_stats.parquet",
        ),
    }


@server.route("/api/players-data")
def _players_data_api():
    global _players_cache
    _players_cache = None   # always rebuild so photo paths are fresh
    try:
        _players_cache = _build_players_payload()
    except Exception as exc:
        return Response(_json.dumps({"error": str(exc), "players": []}),
                        mimetype="application/json", status=500)
    return Response(_players_cache, mimetype="application/json")


# ---------------------------------------------------------------------------
# DUGOUT API — three endpoints used by assets/threejs/dugout.html
# ---------------------------------------------------------------------------

@server.route("/api/dugout-meta")
def _dugout_meta():
    """Return teams, venues, and squad bowlers for the match setup dropdowns."""
    import csv as _csv
    from utils.theme import PSL_2026_TEAMS

    venues = [
        "Gaddafi Stadium, Lahore",
        "National Stadium, Karachi",
        "Rawalpindi Cricket Stadium",
        "Multan Cricket Stadium",
        "Arbab Niaz Stadium, Peshawar",
        "Iqbal Stadium, Faisalabad",
        "Dubai International Cricket Stadium",
        "Sharjah Cricket Stadium",
        "Sheikh Zayed Stadium, Abu Dhabi",
    ]

    bowlers_by_team: dict[str, list[str]] = {}
    try:
        pi_path = _PROJ_ROOT / "data" / "processed" / "player_index_2026_enriched.csv"
        with open(pi_path, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                if str(row.get("is_active", "")).strip().lower() not in ("true", "1", "yes"):
                    continue
                team = str(row.get("current_team_2026", "")).strip()
                role = str(row.get("primary_role", "")).strip()
                nm   = str(row.get("player_name", "")).strip()
                if team and role in ("Bowler", "All-rounder") and nm:
                    bowlers_by_team.setdefault(team, []).append(nm)
        for t in bowlers_by_team:
            bowlers_by_team[t] = sorted(bowlers_by_team[t])
    except Exception:
        pass

    return Response(
        _json.dumps({"teams": PSL_2026_TEAMS, "venues": venues, "bowlers_by_team": bowlers_by_team}),
        mimetype="application/json",
    )


@server.route("/api/dugout-init", methods=["POST"])
def _dugout_init():
    """
    Initialise a match for the Dugout screen.
    Body: {bowling_team, batting_team, venue, innings, target, bowlers:[]}
    Returns: {bowlers, opp_batters, matchups, contingencies, weather}
    """
    import csv as _csv
    import traceback as _tb

    try:
        body          = _flask_request.get_json(force=True) or {}
        bowling_team  = body.get("bowling_team", "Lahore Qalandars")
        batting_team  = body.get("batting_team", "Karachi Kings")
        venue         = body.get("venue", "Gaddafi Stadium, Lahore")
        innings       = int(body.get("innings", 1))
        target        = int(body.get("target", 0))
        sel_bowlers   = body.get("bowlers", [])  # list of names chosen in UI

        pi_path = _PROJ_ROOT / "data" / "processed" / "player_index_2026_enriched.csv"
        ps_path = _PROJ_ROOT / "data" / "processed" / "player_stats.parquet"

        # ── 1. Bowler type map from player_index ──────────────────────────────
        _type_map: dict[str, str] = {}
        try:
            with open(pi_path, newline="", encoding="utf-8") as f:
                for row in _csv.DictReader(f):
                    nm = row.get("player_name", "").strip()
                    if nm:
                        _type_map[nm] = row.get("bowling_style", "") or "Medium Pace"
        except Exception:
            pass

        # ── 2. Career economy from player_stats ───────────────────────────────
        _econ_map: dict[str, float] = {}
        try:
            ps = _pd.read_parquet(ps_path)
            ov = ps[(ps["season"] == 0) & (ps["phase"] == "overall")]
            for _, r in ov.iterrows():
                nm  = str(r.get("player_name", "")).strip()
                eco = r.get("bowl_economy")
                if nm and _pd.notna(eco) and float(eco) > 0:
                    _econ_map[nm] = round(float(eco), 2)
        except Exception:
            pass

        # ── 3. Build bowlers list ─────────────────────────────────────────────
        bowlers = []
        for nm in sel_bowlers:
            bowlers.append({
                "name":    nm,
                "type":    _type_map.get(nm, "Medium Pace"),
                "maxOv":   4,
                "economy": _econ_map.get(nm, 8.0),
                "planned": [],
            })

        # Simple cycling over-allocation (20 overs ÷ n bowlers)
        if bowlers:
            remaining = {b["name"]: 4 for b in bowlers}
            ov_map: dict[int, str] = {}
            bi = 0
            for ov in range(1, 21):
                for _ in range(len(bowlers)):
                    nm = bowlers[bi % len(bowlers)]["name"]
                    if remaining.get(nm, 0) > 0:
                        ov_map[ov] = nm
                        remaining[nm] -= 1
                        bi += 1
                        break
                    bi += 1
            for b in bowlers:
                b["planned"] = sorted(o for o, nm in ov_map.items() if nm == b["name"])

        # ── 4. Opposition batting order ───────────────────────────────────────
        opp_batters = []
        opp_bowling_implications: list[str] = []
        try:
            from engine.opposition_predictor import predict_batting_order
            pred = predict_batting_order(
                team=batting_team, venue=venue,
                our_bowlers=sel_bowlers, season=0,
            )
            opp_bowling_implications = pred.bowling_implications or []
            for b in (pred.predicted_order or []):
                dr = (b.danger_rating or "Low").lower()
                hand = "Left" if "left" in (b.batting_style or "").lower() else "Right"
                opp_batters.append({
                    "name":    b.player_name,
                    "danger":  "high" if dr == "high" else ("med" if "med" in dr else "low"),
                    "hand":    hand,
                    "sr_spin": int(round(float(b.vs_our_spin_sr or b.career_sr or 120))),
                    "sr_pace": int(round(float(b.vs_our_pace_sr or b.career_sr or 120))),
                    "note":    b.key_note or f"#{b.position} — {b.phase_strength}",
                })
        except Exception:
            pass

        # ── 5. Matchups ───────────────────────────────────────────────────────
        matchups = []
        if sel_bowlers and opp_batters:
            try:
                from engine.matchup_engine import get_key_matchups_for_brief
                batter_names = [b["name"] for b in opp_batters]
                notes = get_key_matchups_for_brief(
                    our_bowlers=sel_bowlers,
                    opposition_batters=batter_names,
                    max_notes=6,
                )
                for mn in notes:
                    matchups.append({
                        "bowler": mn.bowler,
                        "batter": mn.batter,
                        "balls":  mn.balls,
                        "sr":     int(round(float(mn.batter_sr))),
                        "edge":   mn.advantage,
                    })
            except Exception:
                pass

        # ── 6. Contingencies ─────────────────────────────────────────────────
        contingencies: list[str] = []
        if opp_bowling_implications:
            contingencies = opp_bowling_implications[:3]
        defaults = [
            "If opening bowler concedes 15+ in over 1, switch second-change immediately.",
            "If a partnership exceeds 40 balls in middle overs, review field and consider a bowling change.",
            "If spinner concedes 14+ before over 12, hold them for overs 13-16 where pitch grips more.",
            "Have a pace option ready for death if RR climbs above 11 after over 15.",
        ]
        for d in defaults:
            if len(contingencies) >= 4:
                break
            contingencies.append(d)

        # ── 7. Weather ────────────────────────────────────────────────────────
        weather = {"spinner_eff": 100, "swing": "Minimal", "dew": "None"}
        try:
            from weather.weather_client import get_weather_for_venue
            from weather.weather_impact import assess_weather_impact
            wr = get_weather_for_venue(venue)
            if wr:
                wi = assess_weather_impact(wr)
                weather["spinner_eff"] = int(round(wi.spinner_penalty * 100))
                bonus_pct = int(round((wi.swing_bonus - 1.0) * 100))
                weather["swing"] = f"+{bonus_pct}%" if bonus_pct > 0 else "Minimal"
                if wi.dew_onset_over:
                    weather["dew"] = f"~Ov {wi.dew_onset_over}"
        except Exception:
            pass

        return Response(_json.dumps({
            "bowlers":      bowlers,
            "opp_batters":  opp_batters,
            "matchups":     matchups,
            "contingencies": contingencies[:5],
            "innings":      innings,
            "target":       target,
            "weather":      weather,
        }), mimetype="application/json")

    except Exception as exc:
        import traceback as _tb
        return Response(
            _json.dumps({"error": str(exc), "trace": _tb.format_exc()}),
            mimetype="application/json", status=500,
        )


@server.route("/api/dugout-situation", methods=["POST"])
def _dugout_situation():
    """
    Rule-based live situation read for the Dugout screen.
    Body: {over, score, wickets, batter1, batter2, part_runs, part_balls,
           bowler, innings, target, balls_in_over}
    Returns: {badge, headline, body, alert}
    """
    try:
        body        = _flask_request.get_json(force=True) or {}
        over        = int(body.get("over", 1))
        score       = int(body.get("score", 0))
        wickets     = int(body.get("wickets", 0))
        part_runs   = int(body.get("part_runs", 0))
        part_balls  = int(body.get("part_balls", 0))
        bowler      = str(body.get("bowler", ""))
        innings     = int(body.get("innings", 1))
        target      = int(body.get("target", 0))
        balls_in_ov = int(body.get("balls_in_over", 0))

        ovsComp = (over - 1) + balls_in_ov / 6.0
        crr     = score / ovsComp if ovsComp > 0 else 0.0
        sr      = int(round((part_runs / part_balls) * 100)) if part_balls > 0 else 0
        phase   = "pp" if over <= 6 else ("death" if over > 16 else "mid")
        b_last  = bowler.split(" ")[-1] if bowler else "Bowler"

        batter1 = str(body.get("batter1", ""))
        batter2 = str(body.get("batter2", ""))

        # ── Claude Sonnet 4.6 tactical read ──────────────────────────────
        claude_result = _claude_dugout(
            over=over, score=score, wickets=wickets,
            part_runs=part_runs, part_balls=part_balls,
            bowler=bowler, innings=innings, target=target,
            balls_in_over=balls_in_ov, phase=phase,
            crr=crr, sr=sr,
            batter1=batter1, batter2=batter2,
        )

        if claude_result:
            return Response(_json.dumps(claude_result), mimetype="application/json")

        # ── Rule-based fallback (if Claude key absent or call fails) ─────
        badge = "info"; headline = ""; body_txt = ""; alert = ""

        if wickets >= 7:
            badge = "crit"
            headline = "Tail exposed — bowl straight at stumps."
            body_txt = f"{wickets} wkts down. Aim at base of stumps. No room outside off. Full and straight."
            alert    = "TAIL: Target stumps. No loose deliveries."

        elif innings == 2 and target > 0 and ovsComp > 0:
            rem  = max(0.0, 20.0 - ovsComp)
            need = target - score
            rrr  = need / rem if rem > 0 else 99.0
            if rrr > 14:
                badge = "crit"
                headline = f"Chase over — {rrr:.1f} RPO needed with {wickets} wkts down."
                body_txt = "Required rate unsustainable. Bowl dry balls and tighten lengths."
                alert    = f"TIGHT: {need} needed in {rem:.1f} overs @ {rrr:.1f} RPO."
            elif rrr > 11:
                badge = "warn"
                headline = f"Chase tight — {rrr:.1f} RPO required, under pressure."
                body_txt = f"Need {need} from {rem:.1f} overs. Bowl dot balls. Limit boundaries."
                alert    = f"REQ RATE: {rrr:.1f}. Dry spell needed immediately."
            elif rrr < 7 and rem > 5:
                badge = "info"
                headline = f"Chase comfortable — {rrr:.1f} RPO needed, maintain control."
                body_txt = f"Need {need} from {rem:.1f} overs. Keep taking wickets."

        if not headline:
            if sr > 160:
                badge = "crit"
                headline = "Partnership accelerating — change now."
                body_txt = f"SR {sr} over {part_balls} balls. Switch bowler. Alter field immediately."
                alert    = f"DANGER: {part_runs}R/{part_balls}B. SR {sr}. Change required."
            elif phase == "death" and crr > 10:
                badge = "warn"
                headline = f"Death pressure — bowl yorkers and cut boundaries."
                body_txt = f"RR {crr:.1f}. Back-of-length to strike batter. {b_last} must execute."
                alert    = f"DEATH: RR {crr:.1f}. {b_last} — yorkers or bowl back of a length."
            elif phase == "pp" and sr < 80 and part_balls > 6:
                badge = "info"
                headline = f"{b_last} — Over {over} on plan. Partnership quiet."
                body_txt = f"PP: {part_runs}R/{part_balls}B. Maintain attacking lengths. Pitch up."
            elif phase == "mid" and crr < 7.0 and over > 8:
                badge = "info"
                headline = "Middle overs — maintain dot pressure."
                body_txt = f"Economy tight at {crr:.1f}. Three dots/over target. Keep spinning."
            else:
                ph_lbl = "Powerplay" if phase == "pp" else ("Death" if phase == "death" else "Middle")
                badge = "info"
                headline = f"Over {over} — {b_last} bowling. {ph_lbl} phase."
                body_txt = f"{ph_lbl}. {part_runs}R/{part_balls}B partnership. CRR {crr:.2f}."

        return Response(_json.dumps({
            "badge": badge, "headline": headline, "body": body_txt, "alert": alert,
        }), mimetype="application/json")

    except Exception as exc:
        return Response(_json.dumps({
            "badge": "info", "headline": "Update failed", "body": str(exc), "alert": "",
        }), mimetype="application/json", status=500)


@server.route("/api/claude-advice", methods=["POST"])
def _claude_advice_api():
    """
    On-demand Claude Sonnet 4.6 tactical advice for any page.
    Body: {context_type, match_context, question}
    Returns: {advice}  — plain text coaching recommendation.
    """
    try:
        body         = _flask_request.get_json(force=True) or {}
        context_type = str(body.get("context_type", "general"))
        match_ctx    = str(body.get("match_context", ""))
        question     = str(body.get("question", ""))

        if not match_ctx:
            return Response(
                _json.dumps({"error": "match_context is required"}),
                mimetype="application/json", status=400,
            )

        advice = _claude_on_demand(context_type, match_ctx, question)
        if advice is None:
            return Response(
                _json.dumps({"advice": "", "error": "Claude unavailable — check ANTHROPIC_API_KEY"}),
                mimetype="application/json",
            )
        return Response(_json.dumps({"advice": advice}), mimetype="application/json")

    except Exception as exc:
        return Response(
            _json.dumps({"advice": "", "error": str(exc)}),
            mimetype="application/json", status=500,
        )


@server.route("/api/coach-analysis", methods=["POST"])
def _coach_analysis_api():
    """
    Full coaching analysis powering all 4 prep room tabs.
    Body: {our_team, opposition, venue, xi_options, bowling_plan, opp_order,
           weather_note, toss_rec}
    Returns: {xi, bowling, opposition}
    """
    try:
        body = _flask_request.get_json(force=True) or {}

        result = _claude_full(
            our_team     = str(body.get("our_team", "")),
            opposition   = str(body.get("opposition", "")),
            venue        = str(body.get("venue", "")),
            xi_options   = body.get("xi_options", []),
            bowling_plan = body.get("bowling_plan", []),
            opp_order    = body.get("opp_order", []),
            weather_note = str(body.get("weather_note", "")),
            toss_rec     = str(body.get("toss_rec", "")),
            weather      = body.get("weather") or None,
        )

        if result is None:
            return Response(
                _json.dumps({"error": "AI unavailable — check GROQ_API_KEY or ANTHROPIC_API_KEY"}),
                mimetype="application/json", status=503,
            )

        return Response(_json.dumps(result), mimetype="application/json")

    except Exception as exc:
        return Response(
            _json.dumps({"error": str(exc)}),
            mimetype="application/json", status=500,
        )


@server.route("/api/coach-eval", methods=["POST"])
def _coach_eval_api():
    """
    Deep head-coach evaluation of all four brief modules.
    Body: {our_team, opposition, venue, xi, bowling_plan, bowler_alloc,
           opp_order, scenarios, toss_rec, weather_note, contingencies}
    Returns: {xi, bowling, opposition, scenarios, verdict} — structured JSON from AI.
    """
    try:
        body = _flask_request.get_json(force=True) or {}

        result = _claude_eval(
            our_team      = str(body.get("our_team", "")),
            opposition    = str(body.get("opposition", "")),
            venue         = str(body.get("venue", "")),
            xi_players    = body.get("xi", []),
            bowling_plan  = body.get("bowling_plan", []),
            bowler_alloc  = body.get("bowler_alloc", []),
            opp_order     = body.get("opp_order", []),
            scenarios     = body.get("scenarios", []),
            toss_rec      = str(body.get("toss_rec", "")),
            weather_note  = str(body.get("weather_note", "")),
            contingencies = body.get("contingencies", []),
        )

        if result is None:
            return Response(
                _json.dumps({"error": "AI unavailable — check GROQ_API_KEY or ANTHROPIC_API_KEY"}),
                mimetype="application/json", status=503,
            )

        return Response(_json.dumps(result), mimetype="application/json")

    except Exception as exc:
        return Response(
            _json.dumps({"error": str(exc)}),
            mimetype="application/json", status=500,
        )


# ---------------------------------------------------------------------------
# DUGOUT — live tactical query (in-match AI ask)
# ---------------------------------------------------------------------------


@server.route("/api/dugout-ask", methods=["POST"])
def _dugout_ask_api():
    """
    Live in-match tactical query from the dugout Tactical Intel Feed.
    Body: {query, over, ball, score, wkts, target, crr, rrr, innings,
           batter1, batter2, current_bowler, phase, bowl_team, bat_team, venue}
    Returns: {answer}
    """
    try:
        body = _flask_request.get_json(force=True) or {}
        query   = str(body.get("query", "")).strip()
        if not query:
            return Response(_json.dumps({"answer": "No query provided."}), mimetype="application/json")

        over    = int(body.get("over", 1))
        ball    = int(body.get("ball", 0))
        score   = int(body.get("score", 0))
        wkts    = int(body.get("wkts", 0))
        target  = int(body.get("target", 0))
        crr     = float(body.get("crr", 0.0))
        rrr     = float(body.get("rrr", 0.0))
        innings = int(body.get("innings", 1))
        batter1 = str(body.get("batter1", ""))
        batter2 = str(body.get("batter2", ""))
        bowler  = str(body.get("current_bowler", ""))
        phase   = str(body.get("phase", "mid"))
        bowl_team = str(body.get("bowl_team", ""))
        bat_team  = str(body.get("bat_team", ""))
        venue     = str(body.get("venue", ""))

        phase_label = {"pp": "Powerplay (overs 1-6)", "mid": "Middle overs (7-16)",
                       "death": "Death overs (17-20)"}.get(phase, phase)

        chase_block = ""
        if innings == 2 and target > 0:
            overs_done = (over - 1) + ball / 6.0
            rem = max(0.0, 20.0 - overs_done)
            need = target - score
            rrr_val = round(need / rem, 2) if rem > 0 else 99.0
            balls_left = max(0, round(rem * 6))
            chase_block = (
                f"- Chase: need {need} off {balls_left} balls (RRR {rrr_val}, CRR {crr:.2f})\n"
            )
        else:
            chase_block = f"- CRR: {crr:.2f}\n" if crr else ""

        batters_block = ""
        if batter1 or batter2:
            batters_block = f"- Batters: {batter1}{', ' + batter2 if batter2 else ''}\n"

        prompt = (
            f"Live T20 dugout query — {bowl_team} bowling vs {bat_team} at {venue}.\n"
            f"Situation: Over {over}.{ball} | Score {score}/{wkts} | Phase: {phase_label}\n"
            f"{chase_block}"
            f"- Current bowler: {bowler if bowler else 'not set'}\n"
            f"{batters_block}"
            f"\nCoach's question: {query}\n\n"
            f"Answer as head coach — direct, specific, max 3 sentences. "
            f"Name specific players and over numbers where relevant."
        )

        from utils.claude_advisor import _call as _ai_call
        answer = _ai_call(prompt, max_tokens=250)
        if answer is None:
            answer = "AI analyst unavailable — check GROQ_API_KEY in your .env file."

        return Response(_json.dumps({"answer": answer}), mimetype="application/json")

    except Exception as exc:
        return Response(
            _json.dumps({"answer": f"Query error: {str(exc)}"}),
            mimetype="application/json", status=500,
        )


# ---------------------------------------------------------------------------
# NAV BAR
# ---------------------------------------------------------------------------


navbar = html.Div(
    className="nav-pill",
    children=[
        # Far-left brand label
        html.Div("INTELLIGENCE DASHBOARD", className="nav-brand-left"),
        # Left spacer (pushes links to centre)
        html.Div(className="nav-spacer"),
        # Nav links — flat horizontal row
        dcc.Link("PREP ROOM", href="/prep",    id="nav-prep",    className="nav-link", refresh=False),
        html.Span("|", className="nav-sep"),
        dcc.Link("DUGOUT",    href="/dugout",  id="nav-dugout",  className="nav-link", refresh=False),
        html.Span("|", className="nav-sep"),
        dcc.Link("PLAYERS",   href="/players", id="nav-players", className="nav-link", refresh=False),
        # Right spacer (balances centre alignment)
        html.Div(className="nav-spacer"),
        # Right group: standby + PSL badge
        html.Div(
            className="nav-right-group",
            children=[
                html.Div(
                    id="live-indicator",
                    className="nav-standby",
                    children=[
                        html.Div(className="nav-standby-dot", id="live-dot"),
                        html.Span("STANDBY", id="live-status-text"),
                    ],
                ),
                html.Div(
                    className="nav-crest-badge",
                    children=[html.Span("PSL", style={
                        "fontFamily": "Orbitron, sans-serif",
                        "fontSize": "8px", "fontWeight": "900",
                        "color": "#1a0a3a", "letterSpacing": "1px",
                    })],
                ),
            ],
        ),
    ],
)

# ---------------------------------------------------------------------------
# ROOT LAYOUT
# ---------------------------------------------------------------------------

_IFRAME_STYLE_VISIBLE = {
    "margin": "0", "padding": "0",
    "height": "calc(100vh - 52px)", "marginTop": "52px",
    "overflow": "hidden", "display": "block",
}
_IFRAME_STYLE_HIDDEN = {**_IFRAME_STYLE_VISIBLE, "display": "none"}
_IFRAME_INNER = {"width": "100%", "height": "100%", "border": "none", "display": "block"}

app.layout = html.Div(
    style={"minHeight": "100vh"},
    className="dash-page-container",
    children=[
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="match-brief-store", storage_type="session"),
        navbar,
        # ── Persistent iframes — never reloaded, just shown/hidden by URL ──
        html.Div(id="iframe-prep", style=_IFRAME_STYLE_VISIBLE, children=[
            html.Iframe(src="/assets/threejs/prep_room_v25.html", style=_IFRAME_INNER),
        ]),
        html.Div(id="iframe-dugout", style=_IFRAME_STYLE_HIDDEN, children=[
            html.Iframe(src="/assets/threejs/dugout_v2.html", style=_IFRAME_INNER),
        ]),
        html.Div(id="iframe-players", style=_IFRAME_STYLE_HIDDEN, children=[
            html.Iframe(src="/assets/threejs/players_v2.html", style=_IFRAME_INNER),
        ]),
        # page_container handles any non-iframe Dash pages (index redirect etc.)
        html.Div(dash.page_container, style={"display": "none"}),
    ],
)

# ---------------------------------------------------------------------------
# NAVBAR ACTIVE STATE CALLBACK
# ---------------------------------------------------------------------------

@app.callback(
    Output("nav-prep",    "className"),
    Output("nav-dugout",  "className"),
    Output("nav-players", "className"),
    Input("url", "pathname"),
)
def update_nav_active(pathname):
    p = pathname or "/"
    if p.startswith("/dugout"):
        return "nav-link", "nav-link active", "nav-link"
    elif p.startswith("/players"):
        return "nav-link", "nav-link", "nav-link active"
    return "nav-link active", "nav-link", "nav-link"


@app.callback(
    Output("iframe-prep",    "style"),
    Output("iframe-dugout",  "style"),
    Output("iframe-players", "style"),
    Input("url", "pathname"),
)
def toggle_iframe_visibility(pathname):
    """Show the matching persistent iframe, hide the others."""
    p = pathname or "/"
    prep    = _IFRAME_STYLE_VISIBLE if (p == "/" or p.startswith("/prep"))    else _IFRAME_STYLE_HIDDEN
    dugout  = _IFRAME_STYLE_VISIBLE if p.startswith("/dugout")                else _IFRAME_STYLE_HIDDEN
    players = _IFRAME_STYLE_VISIBLE if p.startswith("/players")               else _IFRAME_STYLE_HIDDEN
    return prep, dugout, players


# ---------------------------------------------------------------------------
# INDEX / REDIRECT
# ---------------------------------------------------------------------------

# The index page is handled by pages/index.py (redirect to /prep)
# If no pages/index.py, dash will render page_container for "/" which
# defaults to the first registered page.

# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=8050, use_reloader=True)
