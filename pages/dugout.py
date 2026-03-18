# pages/dugout.py
# MODE 2 — Dugout Screen
# One-tap over updates, 5-panel live intelligence feed
# Route: /dugout

from __future__ import annotations

import json
from typing import Optional

import dash
from dash import html, dcc, Input, Output, State, callback, no_update

from utils.theme import (
    BRAND_ORANGE, DARK_BG, DARK_ALT, TEXT_PRIMARY, TEXT_SECONDARY,
    BORDER_COLOR, GREEN, AMBER, RED, STEEL_BLUE,
    PRIORITY_COLORS, DANGER_COLORS, PSL_2026_TEAMS,
)

dash.register_page(__name__, path="/dugout", name="Dugout")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

PSL_TEAMS = PSL_2026_TEAMS

PSL_VENUES = [
    "Gaddafi Stadium, Lahore",
    "National Stadium, Karachi",
    "Rawalpindi Cricket Stadium",
    "Multan Cricket Stadium",
    "Dubai International Cricket Stadium",
    "Sharjah Cricket Stadium",
    "Sheikh Zayed Stadium, Abu Dhabi",
]

def _bowlers_for_team(team: str) -> list[str]:
    """
    Dynamically load bowlers (Bowlers + All-rounders) for a team from player_index_2026_enriched.csv.
    Falls back to an empty list if the file is missing or the team isn't found.
    This replaces the hardcoded TEAM_BOWLERS dict — squad changes mid-tournament
    are reflected automatically without editing source code.
    """
    from pathlib import Path
    import csv
    pi_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "player_index_2026_enriched.csv"
    result = []
    try:
        with open(pi_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                active   = str(row.get("is_active", "True")).strip().lower()
                if active not in ("true", "1", "yes"):
                    continue
                row_team = str(row.get("current_team_2026", "")).strip()
                role     = str(row.get("primary_role", "")).strip()
                if row_team == team and role in ("Bowler", "All-rounder"):
                    name = str(row.get("player_name", "")).strip()
                    if name:
                        result.append(name)
    except Exception:
        pass
    return sorted(result)

# ---------------------------------------------------------------------------
# STYLE CONSTANTS
# ---------------------------------------------------------------------------

CARD = {
    "backgroundColor": DARK_ALT,
    "border": f"1px solid {BORDER_COLOR}",
    "borderRadius": "8px",
    "padding": "14px",
    "height": "100%",
    "boxSizing": "border-box",
    "overflow": "hidden",
}

PANEL_HEADER = {
    "color": TEXT_SECONDARY,
    "fontSize": "0.68rem",
    "fontWeight": "700",
    "letterSpacing": "0.12em",
    "textTransform": "uppercase",
    "marginBottom": "10px",
    "paddingBottom": "6px",
    "borderBottom": f"1px solid {BORDER_COLOR}",
}

LABEL = {
    "color": TEXT_SECONDARY,
    "fontSize": "0.72rem",
    "fontWeight": "600",
    "letterSpacing": "0.08em",
    "textTransform": "uppercase",
    "marginBottom": "5px",
    "display": "block",
}

BTN_BASE = {
    "border": "none",
    "borderRadius": "5px",
    "cursor": "pointer",
    "fontWeight": "700",
    "letterSpacing": "0.06em",
    "padding": "8px 16px",
    "fontSize": "0.80rem",
}

INPUT_STYLE = {
    "backgroundColor": "#1E252D",
    "border": f"1px solid {BORDER_COLOR}",
    "color": TEXT_PRIMARY,
    "borderRadius": "4px",
    "padding": "5px 8px",
    "fontSize": "0.85rem",
    "width": "100%",
}

# ---------------------------------------------------------------------------
# SETUP PANEL
# ---------------------------------------------------------------------------

_setup_panel = html.Div(
    id="dugout-setup",
    style={
        "backgroundColor": DARK_ALT,
        "border": f"1px solid {BORDER_COLOR}",
        "borderLeft": f"3px solid {BRAND_ORANGE}",
        "borderRadius": "8px",
        "padding": "20px",
        "marginBottom": "16px",
    },
    children=[
        html.Div("Match Setup", style={
            "color": BRAND_ORANGE,
            "fontSize": "0.78rem",
            "fontWeight": "700",
            "letterSpacing": "0.12em",
            "textTransform": "uppercase",
            "marginBottom": "14px",
            "paddingBottom": "8px",
            "borderBottom": f"1px solid {BORDER_COLOR}",
        }),
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr 1fr 1fr auto",
                "gap": "14px",
                "alignItems": "end",
            },
            children=[
                html.Div([
                    html.Label("Our Team (Bowling)", style=LABEL),
                    dcc.Dropdown(
                        id="dug-bowling-team",
                        options=[{"label": t, "value": t} for t in PSL_TEAMS],
                        value="Lahore Qalandars",
                        className="psl-dropdown", clearable=False,
                    ),
                ]),
                html.Div([
                    html.Label("Batting Team", style=LABEL),
                    dcc.Dropdown(
                        id="dug-batting-team",
                        options=[{"label": t, "value": t} for t in PSL_TEAMS],
                        value="Karachi Kings",
                        className="psl-dropdown", clearable=False,
                    ),
                ]),
                html.Div([
                    html.Label("Venue", style=LABEL),
                    dcc.Dropdown(
                        id="dug-venue",
                        options=[{"label": v, "value": v} for v in PSL_VENUES],
                        value="Gaddafi Stadium, Lahore",
                        className="psl-dropdown", clearable=False,
                    ),
                ]),
                html.Div([
                    html.Label("Innings / Target", style=LABEL),
                    html.Div(
                        style={"display": "flex", "gap": "6px"},
                        children=[
                            dcc.Dropdown(
                                id="dug-innings",
                                options=[{"label": "1st Innings", "value": 1}, {"label": "2nd Innings", "value": 2}],
                                value=1,
                                className="psl-dropdown", clearable=False,
                                style={"flex": "1"},
                            ),
                            dcc.Input(
                                id="dug-target",
                                type="number", placeholder="Target",
                                min=0, max=350,
                                style={**INPUT_STYLE, "width": "80px"},
                            ),
                        ],
                    ),
                ]),
                html.Button(
                    "START MATCH",
                    id="dug-start-btn",
                    n_clicks=0,
                    style={
                        **BTN_BASE,
                        "backgroundColor": BRAND_ORANGE,
                        "color": "#000",
                        "padding": "10px 20px",
                        "fontSize": "0.82rem",
                        "whiteSpace": "nowrap",
                    },
                ),
            ],
        ),
        # Bowler selector (shown after setup)
        html.Div(
            style={"marginTop": "14px"},
            children=[
                html.Label("Our Bowlers (select 5-7)", style=LABEL),
                dcc.Checklist(
                    id="dug-bowlers",
                    options=[], value=[],
                    style={
                        "display": "flex",
                        "flexWrap": "wrap",
                        "gap": "6px",
                    },
                    inputStyle={"marginRight": "5px", "accentColor": BRAND_ORANGE},
                    labelStyle={
                        "color": TEXT_PRIMARY, "fontSize": "0.80rem",
                        "backgroundColor": "#1E252D",
                        "padding": "4px 10px",
                        "borderRadius": "4px",
                        "border": f"1px solid {BORDER_COLOR}",
                        "cursor": "pointer",
                    },
                ),
            ],
        ),
    ],
)

# ---------------------------------------------------------------------------
# OVER CONTROL BAR
# ---------------------------------------------------------------------------

_over_bar = html.Div(
    id="dugout-controls",
    style={
        "display": "none",
        "backgroundColor": "#0D1219",
        "border": f"1px solid {BORDER_COLOR}",
        "borderRadius": "8px",
        "padding": "12px 16px",
        "marginBottom": "10px",
    },
    children=[
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "auto auto auto auto 1fr auto auto auto auto",
                "gap": "10px",
                "alignItems": "center",
            },
            children=[
                # Prev over
                html.Button(
                    "◄ PREV",
                    id="dug-prev-over",
                    n_clicks=0,
                    style={**BTN_BASE, "backgroundColor": "#1E252D", "color": TEXT_SECONDARY, "padding": "8px 12px"},
                ),
                # Over display
                html.Div(
                    style={"textAlign": "center", "minWidth": "90px"},
                    children=[
                        html.Div("OVER", style={"color": TEXT_SECONDARY, "fontSize": "0.65rem", "letterSpacing": "0.12em"}),
                        html.Div(id="dug-over-display", children="1", style={
                            "color": BRAND_ORANGE, "fontSize": "1.80rem", "fontWeight": "900", "lineHeight": "1",
                        }),
                    ],
                ),
                # Next over
                html.Button(
                    "NEXT ►",
                    id="dug-next-over",
                    n_clicks=0,
                    style={**BTN_BASE, "backgroundColor": BRAND_ORANGE, "color": "#000", "padding": "8px 14px"},
                ),
                # Wicket button
                html.Button(
                    "WICKET FELL",
                    id="dug-wicket-btn",
                    n_clicks=0,
                    style={**BTN_BASE, "backgroundColor": RED, "color": TEXT_PRIMARY, "padding": "8px 14px"},
                ),
                # Spacer
                html.Div(),
                # Score
                html.Div([
                    html.Div("SCORE", style={"color": TEXT_SECONDARY, "fontSize": "0.62rem", "letterSpacing": "0.10em", "textAlign": "center"}),
                    dcc.Input(id="dug-score", type="number", min=0, max=400, value=0,
                              style={**INPUT_STYLE, "width": "80px", "textAlign": "center", "fontSize": "1.1rem", "fontWeight": "700"}),
                ]),
                # Wickets
                html.Div([
                    html.Div("WKTS", style={"color": TEXT_SECONDARY, "fontSize": "0.62rem", "letterSpacing": "0.10em", "textAlign": "center"}),
                    dcc.Dropdown(id="dug-wickets", options=[{"label": str(i), "value": i} for i in range(11)],
                                 value=0, clearable=False, className="psl-dropdown",
                                 style={"width": "65px"}),
                ]),
                # Batters
                html.Div([
                    html.Div("BATTER 1", style={"color": TEXT_SECONDARY, "fontSize": "0.62rem", "letterSpacing": "0.10em"}),
                    dcc.Dropdown(id="dug-batter1", options=[], value=None, placeholder="Batter 1",
                                 className="psl-dropdown", clearable=True,
                                 style={"width": "140px", "fontSize": "0.82rem"}),
                ]),
                html.Div([
                    html.Div("BATTER 2", style={"color": TEXT_SECONDARY, "fontSize": "0.62rem", "letterSpacing": "0.10em"}),
                    dcc.Dropdown(id="dug-batter2", options=[], value=None, placeholder="Batter 2",
                                 className="psl-dropdown", clearable=True,
                                 style={"width": "140px", "fontSize": "0.82rem"}),
                ]),
            ],
        ),
        # Partnership tracker row
        html.Div(
            style={"display": "flex", "gap": "20px", "marginTop": "10px", "alignItems": "center"},
            children=[
                html.Span("PARTNERSHIP:", style={"color": TEXT_SECONDARY, "fontSize": "0.68rem", "letterSpacing": "0.10em"}),
                html.Div([
                    html.Span("Runs ", style={"color": TEXT_SECONDARY, "fontSize": "0.68rem"}),
                    dcc.Input(id="dug-p-runs", type="number", min=0, max=300, value=0,
                              style={**INPUT_STYLE, "width": "65px", "textAlign": "center", "display": "inline-block"}),
                ]),
                html.Div([
                    html.Span("Balls ", style={"color": TEXT_SECONDARY, "fontSize": "0.68rem"}),
                    dcc.Input(id="dug-p-balls", type="number", min=0, max=120, value=0,
                              style={**INPUT_STYLE, "width": "65px", "textAlign": "center", "display": "inline-block"}),
                ]),
                html.Button(
                    "UPDATE",
                    id="dug-update-btn",
                    n_clicks=0,
                    style={**BTN_BASE, "backgroundColor": STEEL_BLUE, "color": TEXT_PRIMARY, "padding": "6px 14px"},
                ),
                html.Span(id="dug-update-status", style={"color": GREEN, "fontSize": "0.75rem"}),
            ],
        ),
    ],
)

# ---------------------------------------------------------------------------
# 5-PANEL LIVE SCREEN
# ---------------------------------------------------------------------------

_live_panels = html.Div(
    id="dugout-live",
    style={"display": "none"},
    children=[
        # Top row: Panel 1 (Bowling Tracker) + Panel 2 (Partnership)
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "3fr 2fr",
                "gap": "10px",
                "marginBottom": "10px",
                "height": "230px",
            },
            children=[
                html.Div(id="panel-bowling-tracker", style=CARD),
                html.Div(id="panel-partnership",     style=CARD),
            ],
        ),
        # Middle row: Panel 3 (Next Batter) + Panel 4 (Situation Read)
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "2fr 3fr",
                "gap": "10px",
                "marginBottom": "10px",
                "height": "200px",
            },
            children=[
                html.Div(id="panel-next-batter",   style=CARD),
                html.Div(id="panel-situation-read", style={
                    **CARD,
                    "display": "flex",
                    "flexDirection": "column",
                    "justifyContent": "center",
                }),
            ],
        ),
        # Bottom: Panel 5 (Weather Bar)
        html.Div(id="panel-weather-bar", style={
            **CARD,
            "height": "60px",
            "display": "flex",
            "alignItems": "center",
            "padding": "0 16px",
        }),
    ],
)

# ---------------------------------------------------------------------------
# STORES
# ---------------------------------------------------------------------------

_stores = html.Div([
    dcc.Store(id="dug-match-store",   storage_type="session"),
    dcc.Store(id="dug-live-store",    storage_type="session"),
    dcc.Store(id="dug-batter-pool",   storage_type="memory"),
    dcc.Store(id="dug-history-store", storage_type="session"),  # list of per-over snapshots
])

# ---------------------------------------------------------------------------
# PAGE LAYOUT
# ---------------------------------------------------------------------------

layout = html.Div(
    style={"backgroundColor": DARK_BG, "minHeight": "100vh", "padding": "14px"},
    children=[
        _stores,
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "16px", "marginBottom": "12px"},
            children=[
                html.H1(
                    "Dugout Screen",
                    style={"color": TEXT_PRIMARY, "fontSize": "1.10rem", "fontWeight": "700",
                           "letterSpacing": "0.04em", "margin": "0"},
                ),
                html.Button(
                    "LOAD MATCH BRIEF",
                    id="dug-load-brief-btn",
                    n_clicks=0,
                    style={
                        **BTN_BASE,
                        "backgroundColor": STEEL_BLUE,
                        "color": TEXT_PRIMARY,
                        "padding": "7px 16px",
                        "fontSize": "0.75rem",
                    },
                ),
                html.Span(id="dug-brief-status", style={"fontSize": "0.80rem"}),
            ],
        ),
        _setup_panel,
        _over_bar,
        _live_panels,
        # Over history — collapsible log of past overs
        html.Div(
            id="dugout-history-section",
            style={"display": "none", "marginTop": "10px"},
            children=[
                html.Div(
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "10px",
                        "marginBottom": "8px",
                    },
                    children=[
                        html.Div("Over History", style={
                            "color": TEXT_SECONDARY,
                            "fontSize": "0.68rem",
                            "fontWeight": "700",
                            "letterSpacing": "0.12em",
                            "textTransform": "uppercase",
                        }),
                        html.Button(
                            "HIDE",
                            id="dug-history-toggle",
                            n_clicks=0,
                            style={
                                **BTN_BASE,
                                "backgroundColor": "#1E252D",
                                "color": TEXT_SECONDARY,
                                "padding": "3px 10px",
                                "fontSize": "0.68rem",
                            },
                        ),
                    ],
                ),
                html.Div(id="panel-over-history"),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# CALLBACK: populate bowlers checklist when bowling team changes
# ---------------------------------------------------------------------------

@callback(
    Output("dug-bowlers", "options"),
    Output("dug-bowlers", "value"),
    Input("dug-bowling-team", "value"),
)
def _populate_bowlers(team: str):
    bowlers = _bowlers_for_team(team) if team else []
    opts = [{"label": b, "value": b} for b in bowlers]
    return opts, bowlers


# ---------------------------------------------------------------------------
# CALLBACK: start match
# ---------------------------------------------------------------------------

@callback(
    Output("dug-match-store",    "data"),
    Output("dug-live-store",     "data"),
    Output("dugout-setup",       "style"),
    Output("dugout-controls",    "style"),
    Output("dugout-live",        "style"),
    Output("dug-over-display",   "children"),
    Input("dug-start-btn",       "n_clicks"),
    State("dug-bowling-team",    "value"),
    State("dug-batting-team",    "value"),
    State("dug-venue",           "value"),
    State("dug-innings",         "value"),
    State("dug-target",          "value"),
    State("dug-bowlers",         "value"),
    prevent_initial_call=True,
)
def _start_match(n_clicks, bowling_team, batting_team, venue, innings, target, bowlers):
    if not bowlers or len(bowlers) < 4:
        return (no_update,) * 6

    target = int(target or 0)

    # Generate bowling plan
    from utils.situation import WeatherImpact
    from engine.bowling_plan import generate_bowling_plan

    try:
        from weather.weather_impact import get_match_weather_impact
        from datetime import datetime
        weather = get_match_weather_impact(venue, datetime.now())
    except Exception:
        weather = WeatherImpact.neutral()

    plan = generate_bowling_plan(
        our_bowlers     = bowlers,
        weather         = weather,
        venue           = venue,
        opposition_team = batting_team,
    )

    # Generate opposition batting order prediction
    from engine.opposition_predictor import predict_batting_order
    opp = predict_batting_order(batting_team, venue, bowlers, season=0)

    # Serialise plan and opposition for store
    plan_data = {
        "overs": [
            {
                "over":            oa.over,
                "primary_bowler":  oa.primary_bowler,
                "backup_bowler":   oa.backup_bowler or "",
                "phase":           oa.phase,
                "reason":          oa.reason,
                "weather_note":    oa.weather_note,
            }
            for oa in plan.overs
        ],
        "bowler_summary": plan.bowler_summary,
        "contingencies":  plan.contingencies,
    }

    opp_data = {
        "team":             opp.team,
        "season":           opp.season,
        "aggressive_opener":opp.aggressive_opener,
        "left_hand_count":  opp.left_hand_count,
        "powerplay_sr":     opp.powerplay_sr,
        "death_sr":         opp.death_sr,
        "vs_spin_economy":  opp.vs_spin_economy,
        "vs_pace_economy":  opp.vs_pace_economy,
        "danger_window":    opp.danger_window,
        "is_estimated":     opp.is_estimated,
        "order": [
            {
                "position":          pb.position,
                "player_name":       pb.player_name,
                "confidence":        pb.confidence,
                "arrival_over_range":pb.arrival_over_range,
                "batting_style":     pb.batting_style,
                "phase_strength":    pb.phase_strength,
                "career_sr":         pb.career_sr,
                "death_sr":          pb.death_sr,
                "vs_our_spin_sr":    pb.vs_our_spin_sr,
                "vs_our_pace_sr":    pb.vs_our_pace_sr,
                "danger_rating":     pb.danger_rating,
                "key_note":          pb.key_note,
            }
            for pb in opp.predicted_order
        ],
        "bowling_implications": opp.bowling_implications,
    }

    weather_data = {
        "spinner_penalty":    weather.spinner_penalty,
        "swing_bonus":        weather.swing_bonus,
        "pace_bounce_bonus":  weather.pace_bounce_bonus,
        "yorker_reliability": weather.yorker_reliability,
        "dl_planning_needed": weather.dl_planning_needed,
        "dew_onset_over":     weather.dew_onset_over or 0,
        "warnings":           weather.warnings,
        "severe_dew":         weather.severe_dew,
    }

    match_store = {
        "bowling_team":  bowling_team,
        "batting_team":  batting_team,
        "venue":         venue,
        "innings":       innings,
        "target":        target,
        "bowlers":       bowlers,
        "plan":          plan_data,
        "opposition":    opp_data,
        "weather":       weather_data,
    }

    live_store = {
        "current_over":        1,
        "current_score":       0,
        "current_wickets":     0,
        "batter1":             "",
        "batter2":             "",
        "partnership_runs":    0,
        "partnership_balls":   0,
        "wickets_this_over":   0,
        "overs_bowled_by":     {b: 0 for b in bowlers},
        "actual_bowlers":      {},   # {"1": "Shaheen Shah Afridi", ...}
    }

    setup_hide = {
        "backgroundColor": DARK_ALT, "border": f"1px solid {BORDER_COLOR}",
        "borderLeft": f"3px solid {BRAND_ORANGE}", "borderRadius": "8px",
        "padding": "20px", "marginBottom": "16px", "display": "none",
    }
    controls_show = {
        "display": "block",
        "backgroundColor": "#0D1219", "border": f"1px solid {BORDER_COLOR}",
        "borderRadius": "8px", "padding": "12px 16px", "marginBottom": "10px",
    }
    live_show = {"display": "block"}

    return match_store, live_store, setup_hide, controls_show, live_show, "1"


# ---------------------------------------------------------------------------
# CALLBACK: advance / reverse over, wicket button
# ---------------------------------------------------------------------------

@callback(
    Output("dug-live-store",    "data",     allow_duplicate=True),
    Output("dug-over-display",  "children", allow_duplicate=True),
    Output("dug-update-status", "children", allow_duplicate=True),
    Output("dug-history-store", "data",     allow_duplicate=True),
    Output("dugout-history-section", "style", allow_duplicate=True),
    Input("dug-next-over",  "n_clicks"),
    Input("dug-prev-over",  "n_clicks"),
    Input("dug-wicket-btn", "n_clicks"),
    Input("dug-update-btn", "n_clicks"),
    State("dug-live-store",    "data"),
    State("dug-match-store",   "data"),
    State("dug-score",         "value"),
    State("dug-wickets",       "value"),
    State("dug-batter1",       "value"),
    State("dug-batter2",       "value"),
    State("dug-p-runs",        "value"),
    State("dug-p-balls",       "value"),
    State("dug-history-store", "data"),
    prevent_initial_call=True,
)
def _update_live(next_n, prev_n, wkt_n, upd_n,
                 live_store, match_store,
                 score, wickets, batter1, batter2, p_runs, p_balls,
                 history_store):
    if live_store is None or match_store is None:
        return no_update, no_update, no_update, no_update, no_update

    from dash import ctx
    triggered = ctx.triggered_id

    ls           = dict(live_store)
    current_over = ls.get("current_over", 1)
    overs_bowled = dict(ls.get("overs_bowled_by", {}))
    actual       = dict(ls.get("actual_bowlers", {}))
    history      = list(history_store or [])
    new_snapshot = None

    if triggered == "dug-next-over":
        if current_over < 20:
            # Snapshot this over BEFORE advancing
            plan_overs = match_store.get("plan", {}).get("overs", [])
            bowler = ""
            for oa in plan_overs:
                if oa["over"] == current_over:
                    bowler = oa["primary_bowler"]
                    overs_bowled[bowler] = overs_bowled.get(bowler, 0) + 1
                    actual[str(current_over)] = bowler
                    break
            new_snapshot = {
                "over":    current_over,
                "score":   int(score or 0),
                "wickets": int(wickets or 0),
                "bowler":  bowler,
                "b1":      str(batter1) if batter1 else "",
                "b2":      str(batter2) if batter2 else "",
                "p_runs":  int(p_runs or 0),
                "p_balls": int(p_balls or 0),
            }
            history.append(new_snapshot)
            current_over += 1
            # Keep partnership balls counting; reset happens on wicket
            ls["partnership_balls"] = int(p_balls or 0) + 6  # rough increment

    elif triggered == "dug-prev-over":
        if current_over > 1:
            # Undo credit for previous over and pop history
            prev = current_over - 1
            plan_overs = match_store.get("plan", {}).get("overs", [])
            for oa in plan_overs:
                if oa["over"] == prev:
                    bowler = oa["primary_bowler"]
                    overs_bowled[bowler] = max(0, overs_bowled.get(bowler, 0) - 1)
                    actual.pop(str(prev), None)
                    break
            # Remove last history snapshot if it matches the reverted over
            if history and history[-1].get("over") == prev:
                history.pop()
            current_over -= 1

    elif triggered == "dug-wicket-btn":
        wkt_val = int(wickets or 0)
        ls["current_wickets"] = min(wkt_val + 1, 10)
        # Reset partnership on wicket
        ls["partnership_runs"]  = 0
        ls["partnership_balls"] = 0

    elif triggered == "dug-update-btn":
        pass  # fields updated below

    # Compute wickets_this_over BEFORE updating current_wickets in ls
    prev_wickets             = ls.get("current_wickets", 0)
    new_wickets              = int(wickets or 0)
    ls["wickets_this_over"]  = max(0, new_wickets - prev_wickets)

    # Always sync manual inputs
    ls["current_over"]      = current_over
    ls["current_score"]     = int(score or 0)
    ls["current_wickets"]   = new_wickets
    ls["batter1"]           = str(batter1) if batter1 else ""
    ls["batter2"]           = str(batter2) if batter2 else ""
    ls["partnership_runs"]  = int(p_runs or 0)
    ls["partnership_balls"] = int(p_balls or 0)
    ls["overs_bowled_by"]   = overs_bowled
    ls["actual_bowlers"]    = actual

    status = "Updated" if triggered == "dug-update-btn" else ""

    # Show history section only when there is at least one completed over
    hist_section_style = (
        {"display": "block", "marginTop": "10px"}
        if history
        else {"display": "none", "marginTop": "10px"}
    )

    return ls, str(current_over), status, history, hist_section_style


# ---------------------------------------------------------------------------
# CALLBACK: render all 5 panels when live store changes
# ---------------------------------------------------------------------------

def _placeholder(msg: str) -> html.Div:
    return html.Div(msg, style={
        "color": "#A0AEC0", "fontSize": "13px", "padding": "20px",
        "textAlign": "center", "height": "100%",
        "display": "flex", "alignItems": "center", "justifyContent": "center",
    })


@callback(
    Output("panel-bowling-tracker", "children"),
    Output("panel-partnership",     "children"),
    Output("panel-next-batter",     "children"),
    Output("panel-situation-read",  "children"),
    Output("panel-weather-bar",     "children"),
    Input("dug-live-store",  "data"),
    State("dug-match-store", "data"),
)
def _render_panels(live_store, match_store):
    if live_store is None or match_store is None:
        return (
            _placeholder("No bowling plan loaded — generate a match brief in Prep Room first, then load it here."),
            _placeholder("Enter current batters and partnership details above, then click Update."),
            _placeholder("Tap WICKET FELL when a wicket falls to see the next batter profile."),
            _placeholder("Situation read will appear once match state is entered above."),
            _placeholder("Weather data will load once venue is set in Prep Room."),
        )
    try:

        ls  = live_store
        ms  = match_store

        current_over    = ls.get("current_over", 1)
        current_score   = ls.get("current_score", 0)
        current_wickets = ls.get("current_wickets", 0)
        batter1         = ls.get("batter1", "") or "—"
        batter2         = ls.get("batter2", "") or "—"
        p_runs          = ls.get("partnership_runs", 0)
        p_balls         = ls.get("partnership_balls", 0)
        overs_bowled    = ls.get("overs_bowled_by", {})
        actual_bowlers  = ls.get("actual_bowlers", {})

        plan_overs      = ms.get("plan", {}).get("overs", [])
        bowler_summary  = ms.get("plan", {}).get("bowler_summary", {})
        contingencies   = ms.get("plan", {}).get("contingencies", [])
        innings         = ms.get("innings", 1)
        target          = ms.get("target", 0)
        batting_team    = ms.get("batting_team", "")
        bowling_team    = ms.get("bowling_team", "")
        venue           = ms.get("venue", "")
        opp_data        = ms.get("opposition", {})
        weather_data    = ms.get("weather", {})

        # Rebuild WeatherImpact from stored data
        from utils.situation import WeatherImpact
        weather = WeatherImpact(
            spinner_penalty    = weather_data.get("spinner_penalty", 1.0),
            swing_bonus        = weather_data.get("swing_bonus", 1.0),
            pace_bounce_bonus  = weather_data.get("pace_bounce_bonus", 1.0),
            yorker_reliability = weather_data.get("yorker_reliability", 1.0),
            dl_planning_needed = weather_data.get("dl_planning_needed", False),
            dew_onset_over     = int(weather_data.get("dew_onset_over", 0) or 0),
            warnings           = weather_data.get("warnings", []),
            # severe_dew is a @property computed from spinner_penalty — not a constructor arg
        )

        # Rebuild BowlingPlan from stored data
        from engine.bowling_plan import BowlingPlan, OverAssignment
        plan_over_list = [
            OverAssignment(
                over           = oa["over"],
                primary_bowler = oa["primary_bowler"],
                backup_bowler  = oa.get("backup_bowler", ""),
                phase          = oa["phase"],
                reason         = oa.get("reason", ""),
                weather_note   = oa.get("weather_note", ""),
            )
            for oa in plan_overs
        ]
        bowling_plan = BowlingPlan(
            overs           = plan_over_list,
            bowler_summary  = {k: list(v) for k, v in bowler_summary.items()},
            contingencies   = contingencies,
        )

        # Derived match state
        balls_left    = max(0, (20 - current_over + 1) * 6)
        required_runs = max(0, target - current_score) if innings == 2 else 0
        rrr = required_runs / (balls_left / 6) if balls_left > 0 and innings == 2 else 0.0
        crr = (current_score / current_over) if current_over > 0 else 0.0

        dew_onset = weather_data.get("dew_onset_over", 0) or 0
        dew_active = (dew_onset > 0 and current_over >= dew_onset)

        phase = (
            "Powerplay" if current_over <= 6
            else "Death"  if current_over >= 16
            else "Middle"
        )

        # Assess partnership — skip engine call when either batter slot is empty
        from engine.partnership_engine import assess_partnership, PartnershipAssessment
        if batter1 != "—" and batter2 != "—" and batter1 and batter2:
            partnership = assess_partnership(
                batter1       = batter1,
                batter2       = batter2,
                current_runs  = p_runs,
                current_balls = p_balls,
            )
        else:
            # Neutral stub so downstream engine calls don't receive None
            partnership = PartnershipAssessment(
                batter1                = batter1 or "—",
                batter2                = batter2 or "—",
                current_runs           = p_runs,
                current_balls          = p_balls,
                current_sr             = 0.0,
                historical_avg_runs    = 0.0,
                historical_avg_balls   = 0.0,
                historical_occurrences = 0,
                danger_level           = "Monitoring",
                danger_trigger         = "No batters selected.",
                break_with_pace_pct    = 50.0,
                break_with_spin_pct    = 35.0,
                break_with_change_pct  = 40.0,
                avg_over_when_broken   = 0.0,
                recommended_action     = "Select batters to enable partnership analysis.",
                confidence             = "Low",
                danger_score           = 0,
                alert_message          = "Partnership: awaiting batter selection.",
                is_historical          = False,
            )

        # Bowler recommendation
        from utils.situation import LiveMatchState
        live_state = LiveMatchState(
            batting_team      = batting_team,
            bowling_team      = bowling_team,
            venue             = venue,
            innings           = innings,
            target            = target,
            bowling_plan      = bowling_plan,
            current_over      = current_over,
            current_score     = current_score,
            current_wickets   = current_wickets,
            current_batter1   = batter1,
            current_batter2   = batter2,
            partnership_runs  = p_runs,
            partnership_balls = p_balls,
            overs_bowled_by   = overs_bowled,
            wickets_this_over = ls.get("wickets_this_over", 0),
        )

        from engine.bowling_recommender import recommend_bowler_this_over
        bowler_rec = recommend_bowler_this_over(bowling_plan, live_state, weather)

        # Opposition prediction (deserialise)
        from engine.opposition_predictor import OppositionBattingPrediction, PredictedBatter
        opp_order = [
            PredictedBatter(
                position          = pb["position"],
                player_name       = pb["player_name"],
                confidence        = pb["confidence"],
                arrival_over_range= pb["arrival_over_range"],
                batting_style     = pb["batting_style"],
                phase_strength    = pb["phase_strength"],
                career_sr         = pb["career_sr"],
                death_sr          = pb["death_sr"],
                vs_our_spin_sr    = pb["vs_our_spin_sr"],
                vs_our_pace_sr    = pb["vs_our_pace_sr"],
                danger_rating     = pb["danger_rating"],
                key_note          = pb["key_note"],
            )
            for pb in opp_data.get("order", [])
        ]
        opp_pred = OppositionBattingPrediction(
            team                = opp_data.get("team", batting_team),
            season              = opp_data.get("season", 0),
            predicted_order     = opp_order,
            bowling_implications= opp_data.get("bowling_implications", []),
            aggressive_opener   = opp_data.get("aggressive_opener", False),
            left_hand_count     = opp_data.get("left_hand_count",    0),
            powerplay_sr        = opp_data.get("powerplay_sr",       0.0),
            vs_spin_economy     = opp_data.get("vs_spin_economy",    0.0),
            vs_pace_economy     = opp_data.get("vs_pace_economy",    0.0),
            death_sr            = opp_data.get("death_sr",           0.0),
            danger_window       = opp_data.get("danger_window",      ""),
            is_estimated        = opp_data.get("is_estimated",       False),
        )

        # Situation read
        from engine.match_intelligence import generate_situation_read
        sit_read = generate_situation_read(
            state        = live_state,
            bowling_plan = bowling_plan,
            partnership  = partnership,
            opposition   = opp_pred,
            weather      = weather,
        )

        # ----------------------------------------------------------------
        # Render panels
        # ----------------------------------------------------------------

        p1 = _render_bowling_tracker(
            plan_overs, current_over, actual_bowlers, overs_bowled, bowler_rec,
        )
        p2 = _render_partnership_panel(partnership, p_runs, p_balls)
        p3 = _render_next_batter(opp_order, current_wickets)
        batting_msg = _live_batting_guidance(
            innings  = innings,
            over     = current_over,
            score    = current_score,
            wickets  = current_wickets,
            target   = target,
            rrr      = live_state.rrr,
            crr      = live_state.crr,
        )
        p4 = _render_situation_read(sit_read, batting_msg)
        p5 = _render_weather_bar(weather, weather_data, innings, current_score, target, current_over)

        return p1, p2, p3, p4, p5

    except Exception as _e:
        err_msg = f"Panel render error — try restarting the match. ({type(_e).__name__}: {_e})"
        return (
            _placeholder(err_msg),
            _placeholder("Enter current batters and partnership details above, then click Update."),
            _placeholder("Tap WICKET FELL when a wicket falls to see the next batter profile."),
            _placeholder("Situation read will appear once match state is entered above."),
            _placeholder("Weather data will load once venue is set in Prep Room."),
        )


# ---------------------------------------------------------------------------
# PANEL RENDER HELPERS
# ---------------------------------------------------------------------------

def _render_bowling_tracker(plan_overs, current_over, actual_bowlers, overs_bowled, bowler_rec) -> html.Div:
    """Panel 1: Bowling plan tracker — 20 over boxes."""

    def _phase_color(phase: str) -> str:
        return {"Powerplay": STEEL_BLUE, "Middle": AMBER, "Death": RED}.get(phase, TEXT_SECONDARY)

    boxes = []
    for oa in plan_overs:
        over_n = oa["over"]
        planned = oa["primary_bowler"].split()[-1]
        phase_col = _phase_color(oa["phase"])

        is_current  = (over_n == current_over)
        is_past     = (over_n < current_over)
        is_future   = (over_n > current_over)

        actual_name = actual_bowlers.get(str(over_n), "")
        actual_last = actual_name.split()[-1] if actual_name else ""
        plan_name   = oa["primary_bowler"]
        deviated    = is_past and actual_name and actual_name != plan_name
        on_plan     = is_past and actual_name and actual_name == plan_name

        # Box style
        bg_color = BRAND_ORANGE if is_current else ("#1A1F26" if is_past else "#111820")
        border_c = BRAND_ORANGE if is_current else (GREEN if on_plan else (RED if deviated else BORDER_COLOR))
        text_col = "#000" if is_current else (TEXT_SECONDARY if is_future else TEXT_PRIMARY)

        indicator = ""
        if on_plan:      indicator = "✓"
        elif deviated:   indicator = "✗"

        boxes.append(
            html.Div(
                style={
                    "backgroundColor": bg_color,
                    "border": f"1px solid {border_c}",
                    "borderTop": f"2px solid {phase_col if not is_current else '#000'}",
                    "borderRadius": "4px",
                    "padding": "4px 5px",
                    "textAlign": "center",
                    "position": "relative",
                },
                children=[
                    html.Div(
                        str(over_n),
                        style={"color": text_col, "fontSize": "0.60rem", "fontWeight": "700"},
                    ),
                    html.Div(
                        planned,
                        style={"color": text_col, "fontSize": "0.68rem", "fontWeight": "600" if is_current else "400",
                               "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"},
                    ),
                    html.Div(
                        indicator,
                        style={
                            "color": GREEN if on_plan else RED,
                            "fontSize": "0.65rem",
                            "fontWeight": "900",
                        },
                    ),
                ],
            )
        )

    # Deviation alert
    alert = None
    if bowler_rec.warning:
        alert = html.Div(
            style={
                "display": "flex", "gap": "6px", "alignItems": "flex-start",
                "padding": "6px 10px",
                "backgroundColor": "#2A1800",
                "border": f"1px solid {AMBER}",
                "borderRadius": "4px",
                "marginTop": "8px",
            },
            children=[
                html.Span("⚠", style={"color": AMBER}),
                html.Span(bowler_rec.warning, style={"color": AMBER, "fontSize": "0.75rem"}),
            ],
        )

    # Bowler overs remaining
    summary_chips = []
    # Build chips for all bowlers
    for bowler, used in overs_bowled.items():
        remaining = max(0, 4 - used)
        chip_col  = GREEN if remaining >= 3 else AMBER if remaining >= 1 else RED
        summary_chips.append(
            html.Span(
                f"{bowler.split()[-1]} ({remaining})",
                style={
                    "color": chip_col,
                    "fontSize": "0.68rem",
                    "padding": "2px 7px",
                    "borderRadius": "3px",
                    "backgroundColor": f"{chip_col}15",
                    "border": f"1px solid {chip_col}50",
                    "whiteSpace": "nowrap",
                },
            )
        )

    return html.Div(
        style={"height": "100%", "display": "flex", "flexDirection": "column"},
        children=[
            html.Div("Bowling Plan Tracker", style=PANEL_HEADER),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(10, 1fr)",
                    "gap": "3px",
                    "marginBottom": "8px",
                    "flex": "1",
                },
                children=boxes,
            ),
            html.Div(
                style={"display": "flex", "gap": "5px", "flexWrap": "wrap"},
                children=summary_chips,
            ),
            *([alert] if alert else []),
        ],
    )


def _render_partnership_panel(partnership, p_runs, p_balls) -> html.Div:
    """Panel 2: Partnership alert with danger meter."""
    danger_col = DANGER_COLORS.get(partnership.danger_level, TEXT_SECONDARY)
    score_pct  = min(100, partnership.danger_score)

    # SR display
    p_sr = round(p_runs / p_balls * 100, 1) if p_balls > 0 else 0.0

    # Progress bar colour gradient (green → amber → red)
    bar_col = GREEN if score_pct < 40 else AMBER if score_pct < 70 else RED

    # How broken historically
    break_method = ""
    if partnership.break_with_pace_pct >= 50:
        break_method = f"Pace breaks them {partnership.break_with_pace_pct:.0f}% of the time"
    elif partnership.break_with_spin_pct >= 50:
        break_method = f"Spin breaks them {partnership.break_with_spin_pct:.0f}% of the time"
    elif partnership.break_with_change_pct >= 40:
        break_method = f"Bowling change breaks them {partnership.break_with_change_pct:.0f}%"

    return html.Div(
        style={"height": "100%", "display": "flex", "flexDirection": "column"},
        children=[
            html.Div("Partnership Alert", style=PANEL_HEADER),
            # Names
            html.Div(
                style={"display": "flex", "gap": "8px", "marginBottom": "10px", "alignItems": "center"},
                children=[
                    html.Span(
                        partnership.batter1 or "—",
                        style={"color": TEXT_PRIMARY, "fontWeight": "700", "fontSize": "0.88rem"},
                    ),
                    html.Span("&", style={"color": TEXT_SECONDARY, "fontSize": "0.80rem"}),
                    html.Span(
                        partnership.batter2 or "—",
                        style={"color": TEXT_PRIMARY, "fontWeight": "700", "fontSize": "0.88rem"},
                    ),
                ],
            ),
            # Stats row
            html.Div(
                style={"display": "flex", "gap": "16px", "marginBottom": "10px"},
                children=[
                    html.Div([
                        html.Div(str(p_runs), style={"color": TEXT_PRIMARY, "fontSize": "1.30rem", "fontWeight": "900"}),
                        html.Div("RUNS", style={"color": TEXT_SECONDARY, "fontSize": "0.62rem", "letterSpacing": "0.1em"}),
                    ]),
                    html.Div([
                        html.Div(str(p_balls), style={"color": TEXT_PRIMARY, "fontSize": "1.30rem", "fontWeight": "900"}),
                        html.Div("BALLS", style={"color": TEXT_SECONDARY, "fontSize": "0.62rem", "letterSpacing": "0.1em"}),
                    ]),
                    html.Div([
                        html.Div(f"{p_sr:.0f}", style={"color": TEXT_PRIMARY, "fontSize": "1.30rem", "fontWeight": "900"}),
                        html.Div("SR", style={"color": TEXT_SECONDARY, "fontSize": "0.62rem", "letterSpacing": "0.1em"}),
                    ]),
                ],
            ),
            # Danger level badge
            html.Div(
                style={
                    "display": "inline-flex",
                    "alignItems": "center",
                    "gap": "6px",
                    "marginBottom": "8px",
                },
                children=[
                    html.Div(
                        style={"width": "8px", "height": "8px", "borderRadius": "50%", "backgroundColor": danger_col},
                    ),
                    html.Span(
                        partnership.danger_level.upper(),
                        style={"color": danger_col, "fontSize": "0.80rem", "fontWeight": "800", "letterSpacing": "0.10em"},
                    ),
                ],
            ),
            # Progress bar
            html.Div(
                style={"backgroundColor": "#1E252D", "borderRadius": "4px", "height": "6px", "marginBottom": "8px"},
                children=[
                    html.Div(
                        style={
                            "width": f"{score_pct}%",
                            "backgroundColor": bar_col,
                            "height": "6px",
                            "borderRadius": "4px",
                            "transition": "width 0.4s ease",
                        },
                    ),
                ],
            ),
            # Recommendation
            html.Div(
                partnership.recommended_action,
                style={"color": TEXT_SECONDARY, "fontSize": "0.75rem", "marginBottom": "4px"},
            ),
            html.Div(
                break_method,
                style={"color": AMBER if break_method else "transparent", "fontSize": "0.73rem", "fontWeight": "600"},
            ),
        ],
    )


def _render_next_batter(opp_order: list, current_wickets: int) -> html.Div:
    """Panel 3: Next batter incoming."""
    # Next batter = position (wickets + 2) i.e. if 1 wicket fell, position 3 is next
    next_pos = current_wickets + 2
    next_batter = None
    for pb in opp_order:
        if pb.position == next_pos:
            next_batter = pb
            break
    # Also find who's currently batting (positions ≤ wickets+1 who aren't out)
    on_deck_pos = current_wickets + 3
    on_deck = None
    for pb in opp_order:
        if pb.position == on_deck_pos:
            on_deck = pb
            break

    if next_batter is None:
        return html.Div(
            style={"height": "100%", "display": "flex", "flexDirection": "column"},
            children=[
                html.Div("Next Batter Incoming", style=PANEL_HEADER),
                html.Div("—", style={"color": TEXT_SECONDARY, "fontSize": "1.2rem"}),
            ],
        )

    danger_col = DANGER_COLORS.get(next_batter.danger_rating, TEXT_SECONDARY)

    return html.Div(
        style={"height": "100%", "display": "flex", "flexDirection": "column"},
        children=[
            html.Div("Next Batter Incoming", style=PANEL_HEADER),
            # Name + danger
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "marginBottom": "8px"},
                children=[
                    html.Div([
                        html.Div(
                            f"#{next_pos}",
                            style={"color": TEXT_SECONDARY, "fontSize": "0.68rem"},
                        ),
                        html.Div(
                            next_batter.player_name,
                            style={"color": TEXT_PRIMARY, "fontSize": "1.05rem", "fontWeight": "800"},
                        ),
                        html.Div(
                            next_batter.phase_strength,
                            style={"color": TEXT_SECONDARY, "fontSize": "0.72rem"},
                        ),
                    ]),
                    html.Span(
                        next_batter.danger_rating.upper(),
                        style={
                            "color": danger_col,
                            "fontSize": "0.72rem",
                            "fontWeight": "800",
                            "padding": "3px 8px",
                            "borderRadius": "4px",
                            "backgroundColor": f"{danger_col}20",
                            "border": f"1px solid {danger_col}",
                        },
                    ),
                ],
            ),
            # Stats
            html.Div(
                style={"display": "flex", "gap": "14px", "marginBottom": "8px"},
                children=[
                    html.Div([
                        html.Div(f"{next_batter.career_sr:.0f}", style={"color": TEXT_PRIMARY, "fontWeight": "700", "fontSize": "0.95rem"}),
                        html.Div("PSL SR", style={"color": TEXT_SECONDARY, "fontSize": "0.60rem", "letterSpacing": "0.08em"}),
                    ]),
                    html.Div([
                        html.Div(f"{next_batter.vs_our_spin_sr:.0f}", style={"color": TEXT_PRIMARY, "fontWeight": "700", "fontSize": "0.95rem"}),
                        html.Div("vs Spin", style={"color": TEXT_SECONDARY, "fontSize": "0.60rem", "letterSpacing": "0.08em"}),
                    ]),
                    html.Div([
                        html.Div(f"{next_batter.vs_our_pace_sr:.0f}", style={"color": TEXT_PRIMARY, "fontWeight": "700", "fontSize": "0.95rem"}),
                        html.Div("vs Pace", style={"color": TEXT_SECONDARY, "fontSize": "0.60rem", "letterSpacing": "0.08em"}),
                    ]),
                    html.Div([
                        html.Div(
                            next_batter.batting_style.split("-")[0],
                            style={"color": TEXT_PRIMARY, "fontWeight": "700", "fontSize": "0.95rem"},
                        ),
                        html.Div("Hand", style={"color": TEXT_SECONDARY, "fontSize": "0.60rem", "letterSpacing": "0.08em"}),
                    ]),
                ],
            ),
            # Key note
            html.Div(
                next_batter.key_note,
                style={
                    "color": TEXT_SECONDARY, "fontSize": "0.73rem",
                    "fontStyle": "italic",
                    "borderLeft": f"3px solid {danger_col}",
                    "paddingLeft": "8px",
                },
            ),
        ],
    )


def _live_batting_guidance(
    innings: int,
    over: int,
    score: int,
    wickets: int,
    target: int,
    rrr: float,
    crr: float,
) -> str | None:
    """
    Rule-based live batting scenario guidance for the dugout.
    Returns a single plain-English sentence for the coaching staff, or None if no
    specific guidance applies (match is on track — no extra nudge needed).

    Covers five live scenarios that pre-match batting cards cannot predict:
      1. Collapse mid-innings (3+ wickets inside 10 overs in innings 1)
      2. Chasing comfortably (innings 2, RRR <= 6)
      3. Hard chase accelerate (innings 2, RRR >= 12 and overs 11-15)
      4. Death push (over >= 16, innings 1 or 2)
      5. DLS risk front-load (weather.dl_planning_needed covered in sit_read — not duplicated here)
    """
    if innings == 1:
        # Collapse scenario: 3+ wickets in overs 1-10 (powerplay / early middle)
        if wickets >= 3 and over <= 10:
            return (
                f"Batting: {wickets} wickets in overs 1-{over} — rebuild phase. "
                f"Anchor: minimise dot balls, rotate strike, no slog before over 14."
            )
        # Death push
        if over >= 16 and wickets <= 5:
            return (
                f"Batting: Death push — {wickets} down, overs 16-20. "
                f"Finisher role: clear boundaries, target spinners / short balls."
            )
        if over >= 16 and wickets >= 6:
            return (
                f"Batting: Tail in hand over 16 — protect key batter, back-foot boundaries only."
            )

    elif innings == 2 and target > 0:
        balls_left = max(1, (20 - over) * 6)
        runs_needed = max(0, target - score)

        # Chase near-complete
        if runs_needed <= 20 and over >= 17:
            return f"Chase: {runs_needed} needed off {balls_left} balls — close it out, no risk shots."

        # Comfortable chase
        if rrr > 0 and rrr <= 6.5 and over >= 6:
            return (
                f"Chase: RRR {rrr:.1f} — on track. Stay positive; don't let rate drift above 8."
            )

        # Hard chase — must accelerate
        if rrr >= 12.0 and 10 <= over <= 15:
            return (
                f"Chase: RRR {rrr:.1f} — acceleration phase. "
                f"Take calculated risks overs {over+1}-16; keep wickets for the final push."
            )

        # Urgent chase — almost impossible without boundaries every over
        if rrr >= 15.0:
            return (
                f"Chase: RRR {rrr:.1f} — boundary or bust. "
                f"Target the first bad ball each over — no dot ball tolerance."
            )

    return None


def _render_situation_read(sit_read, live_batting_msg: str | None = None) -> html.Div:
    """Panel 4: Match situation read — the most prominent panel."""
    if sit_read is None:
        sit_read = type("_SR", (), {
            "priority": "INFO", "message": "Awaiting match data.",
            "action_needed": False, "detail": "",
        })()
    priority_col = PRIORITY_COLORS.get(sit_read.priority or "INFO", TEXT_PRIMARY)

    return html.Div(
        style={"height": "100%", "display": "flex", "flexDirection": "column", "justifyContent": "center"},
        children=[
            html.Div("Match Situation Read", style=PANEL_HEADER),
            # Priority badge
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "10px", "marginBottom": "12px"},
                children=[
                    html.Div(
                        sit_read.priority,
                        style={
                            "color": "#000" if sit_read.priority == "INFO" else priority_col,
                            "backgroundColor": priority_col,
                            "fontSize": "0.70rem",
                            "fontWeight": "900",
                            "letterSpacing": "0.12em",
                            "padding": "3px 10px",
                            "borderRadius": "3px",
                        },
                    ),
                    html.Span(
                        "ACTION NEEDED" if sit_read.action_needed else "",
                        style={"color": AMBER, "fontSize": "0.68rem", "fontWeight": "700", "letterSpacing": "0.10em"},
                    ),
                ],
            ),
            # Main message — the one sentence for the coach
            html.Div(
                sit_read.message,
                style={
                    "color": priority_col,
                    "fontSize": "1.05rem",
                    "fontWeight": "700",
                    "lineHeight": "1.4",
                    "marginBottom": "12px",
                },
            ),
            # Detail for analyst
            html.Div(
                sit_read.detail,
                style={
                    "color": TEXT_SECONDARY,
                    "fontSize": "0.75rem",
                    "lineHeight": "1.5",
                    "borderTop": f"1px solid {BORDER_COLOR}",
                    "paddingTop": "8px",
                },
            ),
            # Live batting guidance (rule-based, scenario-aware)
            *(
                [html.Div(
                    [
                        html.Span("BATTING GUIDE  ", style={
                            "color": STEEL_BLUE, "fontSize": "0.65rem",
                            "fontWeight": "800", "letterSpacing": "0.10em",
                        }),
                        html.Span(live_batting_msg, style={
                            "color": TEXT_PRIMARY, "fontSize": "0.74rem",
                        }),
                    ],
                    style={
                        "marginTop": "10px",
                        "borderTop": f"1px solid {BORDER_COLOR}",
                        "paddingTop": "8px",
                    },
                )]
                if live_batting_msg else []
            ),
        ],
    )


def _render_weather_bar(weather, weather_data, innings, score, target, current_over) -> html.Div:
    """Panel 5: Weather + D/L status bar."""
    dew_onset   = weather_data.get("dew_onset_over", 0) or 0
    dew_active  = (dew_onset > 0 and current_over >= dew_onset)
    dew_label   = f"DEW: ACTIVE (since over {dew_onset})" if dew_active else (
                  f"DEW: from over {dew_onset}"           if dew_onset > 0 else "DEW: None"
    )
    dew_col     = RED if dew_active else (AMBER if dew_onset > 0 else GREEN)

    items = [
        html.Span(f"Spinner eff: {int(weather.spinner_penalty * 100)}%",
                  style={"color": TEXT_SECONDARY, "fontSize": "0.75rem"}),
        html.Span("  |  ", style={"color": BORDER_COLOR}),
        html.Span(f"Swing: +{int((weather.swing_bonus - 1) * 100)}%",
                  style={"color": TEXT_SECONDARY, "fontSize": "0.75rem"}),
        html.Span("  |  ", style={"color": BORDER_COLOR}),
        html.Span(
            dew_label,
            style={"color": dew_col, "fontSize": "0.75rem", "fontWeight": "700"},
        ),
    ]

    # D/L par if innings 2 and rain risk
    if innings == 2 and target > 0 and weather.dl_planning_needed:
        balls_left = max(0, (20 - current_over + 1) * 6)
        required   = max(0, target - score)
        rrr        = required / (balls_left / 6) if balls_left > 0 else 0.0
        crr        = (score / current_over) if current_over > 0 else 0.0
        par_col    = RED if rrr > crr * 1.15 else GREEN
        items += [
            html.Span("  |  ", style={"color": BORDER_COLOR}),
            html.Span(
                f"D/L  Need: {required} off {balls_left // 6}.{balls_left % 6} ov  RRR: {rrr:.1f}  CRR: {crr:.1f}",
                style={"color": par_col, "fontSize": "0.75rem", "fontWeight": "700"},
            ),
        ]
    elif innings == 2 and target > 0:
        balls_left = max(0, (20 - current_over + 1) * 6)
        required   = max(0, target - score)
        rrr        = required / (balls_left / 6) if balls_left > 0 else 0.0
        crr        = (score / current_over) if current_over > 0 else 0.0
        par_col    = RED if rrr > crr * 1.20 else (AMBER if rrr > crr * 1.05 else GREEN)
        items += [
            html.Span("  |  ", style={"color": BORDER_COLOR}),
            html.Span(
                f"Need: {required} off {balls_left // 6}.{balls_left % 6} ov  RRR: {rrr:.1f}  CRR: {crr:.1f}",
                style={"color": par_col, "fontSize": "0.75rem", "fontWeight": "700"},
            ),
        ]

    return html.Div(
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "0",
            "flexWrap": "wrap",
            "height": "100%",
        },
        children=items,
    )


# ---------------------------------------------------------------------------
# CALLBACK: render over-history panel
# ---------------------------------------------------------------------------

@callback(
    Output("panel-over-history", "children"),
    Input("dug-history-store",   "data"),
    State("dug-match-store",     "data"),
)
def _render_history_panel(history, match_store):
    if not history:
        return html.Div()

    plan_overs = (match_store or {}).get("plan", {}).get("overs", [])
    plan_map   = {oa["over"]: oa for oa in plan_overs}

    cards = []
    for snap in reversed(history):   # most recent first
        over    = snap.get("over", "?")
        score   = snap.get("score", 0)
        wickets = snap.get("wickets", 0)
        bowler  = snap.get("bowler", "—") or "—"
        b1      = snap.get("b1", "") or "—"
        b2      = snap.get("b2", "") or "—"
        p_runs  = snap.get("p_runs", 0)
        p_balls = snap.get("p_balls", 0)

        # Check plan deviation
        planned = plan_map.get(over, {}).get("primary_bowler", "")
        deviated = planned and bowler != planned and bowler != "—"
        bowler_col = RED if deviated else TEXT_PRIMARY
        bowler_label = f"{bowler.split()[-1] if bowler != '—' else '—'}"
        if deviated:
            bowler_label += f" [plan: {planned.split()[-1]}]"

        # Phase color
        phase = plan_map.get(over, {}).get("phase", "")
        phase_col = {"Powerplay": STEEL_BLUE, "Middle": AMBER, "Death": RED}.get(phase, BORDER_COLOR)

        p_sr = f"{p_runs / p_balls * 100:.0f}" if p_balls > 0 else "—"

        cards.append(
            html.Div(
                style={
                    "backgroundColor": DARK_ALT,
                    "border": f"1px solid {BORDER_COLOR}",
                    "borderTop": f"2px solid {phase_col}",
                    "borderRadius": "5px",
                    "padding": "8px 10px",
                    "minWidth": "130px",
                    "flex": "0 0 auto",
                },
                children=[
                    html.Div(
                        f"Over {over}",
                        style={"color": BRAND_ORANGE, "fontSize": "0.72rem", "fontWeight": "800", "marginBottom": "4px"},
                    ),
                    html.Div(
                        f"{score}/{wickets}",
                        style={"color": TEXT_PRIMARY, "fontSize": "1.05rem", "fontWeight": "900", "letterSpacing": "0.04em"},
                    ),
                    html.Div(
                        bowler_label,
                        style={"color": bowler_col, "fontSize": "0.70rem", "marginTop": "2px",
                               "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis", "maxWidth": "130px"},
                    ),
                    html.Div(
                        f"{b1.split()[-1] if b1 != '—' else '—'} & {b2.split()[-1] if b2 != '—' else '—'}",
                        style={"color": TEXT_SECONDARY, "fontSize": "0.68rem", "marginTop": "2px"},
                    ),
                    html.Div(
                        f"Pship: {p_runs}/{p_balls} ({p_sr} SR)",
                        style={"color": TEXT_SECONDARY, "fontSize": "0.65rem", "marginTop": "2px"},
                    ),
                ],
            )
        )

    return html.Div(
        style={
            "display": "flex",
            "gap": "8px",
            "overflowX": "auto",
            "paddingBottom": "6px",
        },
        children=cards,
    )


# ---------------------------------------------------------------------------
# CALLBACK: toggle over-history visibility
# ---------------------------------------------------------------------------

@callback(
    Output("panel-over-history", "style"),
    Output("dug-history-toggle", "children"),
    Input("dug-history-toggle",  "n_clicks"),
    prevent_initial_call=True,
)
def _toggle_history(n_clicks):
    if n_clicks % 2 == 1:
        return {"display": "none"}, "SHOW"
    return {}, "HIDE"


# ---------------------------------------------------------------------------
# CALLBACK: Load Match Brief from shared store
# ---------------------------------------------------------------------------

@callback(
    Output("dug-brief-status",    "children"),
    Output("dug-bowling-team",    "value"),
    Output("dug-batting-team",    "value"),
    Output("dug-venue",           "value"),
    Output("dug-batter-pool",     "data"),
    Input("dug-load-brief-btn",   "n_clicks"),
    State("match-brief-store",    "data"),
    prevent_initial_call=True,
)
def _load_brief(n_clicks, brief_data):
    if not brief_data:
        return (
            html.Span("No brief found — please generate one in Prep Room first.",
                      style={"color": AMBER}),
            no_update, no_update, no_update, no_update,
        )
    required_keys = {"our_team", "opposition", "venue", "squad"}
    missing = required_keys - set(brief_data.keys())
    if missing:
        return (
            html.Span(
                f"Brief is incomplete (missing: {', '.join(sorted(missing))}) — "
                f"please regenerate the brief in Prep Room.",
                style={"color": RED},
            ),
            no_update, no_update, no_update, no_update,
        )
    our_team   = brief_data.get("our_team", "")
    opposition = brief_data.get("opposition", "")
    venue      = brief_data.get("venue", "")
    squad      = brief_data.get("squad", [])

    # Load batting team (opposition) players for batter dropdowns
    from pathlib import Path
    import pandas as pd
    pi_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "player_index_2026_enriched.csv"
    batting_players = []
    try:
        pi = pd.read_csv(pi_path)
        batting_players = pi[pi["current_team_2026"] == opposition]["player_name"].sort_values().tolist()
    except Exception:
        pass
    if not batting_players:
        batting_players = squad  # fallback to our squad if opp not found

    msg = html.Span(
        f"Brief loaded — {our_team} vs {opposition} at {venue}",
        style={"color": GREEN},
    )
    return msg, our_team, opposition, venue, batting_players


# ---------------------------------------------------------------------------
# CALLBACK: Populate batter dropdowns from batter pool
# ---------------------------------------------------------------------------

@callback(
    Output("dug-batter1", "options"),
    Output("dug-batter2", "options"),
    Input("dug-batter-pool", "data"),
    Input("dug-batting-team", "value"),
)
def _populate_batter_dropdowns(pool, batting_team):
    if pool:
        players = pool
    else:
        # Load from player_index for selected batting team
        from pathlib import Path
        import pandas as pd
        pi_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "player_index_2026_enriched.csv"
        players = []
        try:
            pi = pd.read_csv(pi_path)
            players = pi[pi["current_team_2026"] == (batting_team or "")]["player_name"].sort_values().tolist()
        except Exception:
            pass
    opts = [{"label": p, "value": p} for p in players]
    return opts, opts
