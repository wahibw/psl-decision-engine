# pages/prep_room.py
# MODE 1 — Match Prep Room
# Input: team, opposition, venue, date/time, squad
# Output: 7 sections + PDF download button
# Route: /prep

from __future__ import annotations

import base64
import json
import os
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import dash
from dash import html, dcc, Input, Output, State, callback, ctx, no_update
import dash_bootstrap_components as dbc

import re

from utils.theme import (
    BRAND_ORANGE, BRAND_ACCENT, DARK_BG, DARK_ALT,
    TEXT_PRIMARY, TEXT_SECONDARY, BORDER_COLOR,
    GREEN, AMBER, RED, STEEL_BLUE,
    PRIORITY_COLORS, DANGER_COLORS, CONFIDENCE_COLORS,
    PSL_TEAM_COLORS, PSL_2026_TEAMS,
)

dash.register_page(__name__, path="/prep", name="Prep Room")

PROJ_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# STATIC DATA — teams, venues, squad lists
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

def _load_active_players() -> pd.DataFrame:
    """Load player_index_2026_enriched.csv and return only active players."""
    pi_path = PROJ_ROOT / "data" / "processed" / "player_index_2026_enriched.csv"
    try:
        pi = pd.read_csv(pi_path)
        if "is_active" in pi.columns:
            return pi[pi["is_active"] == True].copy()
        return pi.copy()
    except Exception as e:
        import warnings
        warnings.warn(
            f"Failed to load player index from {pi_path}: {e}. "
            f"Squad dropdowns will be empty.",
            UserWarning, stacklevel=2,
        )
        return pd.DataFrame(columns=["player_name", "current_team_2026", "data_tier"])


def _players_for_team(team: str) -> list[str]:
    """All active players whose current_team_2026 matches the given team."""
    pi = _load_active_players()
    if "current_team_2026" not in pi.columns:
        return []
    return pi[pi["current_team_2026"] == team]["player_name"].tolist()


def _all_active_players() -> list[str]:
    """All active players, sorted alphabetically."""
    pi = _load_active_players()
    return sorted(pi["player_name"].tolist())


def _build_display_map() -> dict[str, str]:
    """Return {player_name|team: display_label}. Duplicates get '(Team)' suffix."""
    pi_path = PROJ_ROOT / "data" / "processed" / "player_index_2026_enriched.csv"
    try:
        pi = pd.read_csv(pi_path)
        dup_names = set(pi["player_name"][pi["player_name"].duplicated(keep=False)])
        result = {}
        for _, row in pi.iterrows():
            name = str(row["player_name"])
            team = str(row.get("current_team_2026", "") or "")
            key = f"{name}|{team}"
            result[key] = f"{name} ({team})" if name in dup_names and team else name
        return result
    except Exception as e:
        import warnings
        warnings.warn(
            f"Failed to build player display map: {e}. Player labels may be missing.",
            UserWarning, stacklevel=2,
        )
        return {}


def _disambiguate(player_name: str, team: str) -> str:
    """Return display label with '(Team)' suffix if the name is a duplicate."""
    dm = _build_display_map()
    return dm.get(f"{player_name}|{team}", player_name)


def _strip_disambiguation(name: str) -> str:
    """Strip '(Team)' suffix to get the raw player_name for engine lookups."""
    return re.sub(r"\s*\([^)]+\)\s*$", "", name).strip()


def _get_tier_badge(player_name: str, pi_df: pd.DataFrame) -> list:
    """Return tier badge element(s) for insertion next to a player name."""
    clean = _strip_disambiguation(player_name)
    if pi_df.empty:
        return []
    row = pi_df[pi_df["player_name"] == clean]
    if row.empty:
        return []
    tier = int(row.iloc[0].get("data_tier", 1) or 1)
    if tier == 2:
        return [html.Span(
            "Limited PSL data",
            style={
                "color": TEXT_SECONDARY, "fontSize": "0.62rem", "fontWeight": "600",
                "backgroundColor": f"{TEXT_SECONDARY}22", "padding": "1px 5px",
                "borderRadius": "3px", "marginLeft": "5px", "verticalAlign": "middle",
            },
        )]
    if tier == 3:
        return [html.Span(
            "T20 estimate",
            style={
                "color": AMBER, "fontSize": "0.62rem", "fontWeight": "700",
                "backgroundColor": f"{AMBER}22", "padding": "1px 5px",
                "borderRadius": "3px", "marginLeft": "5px", "verticalAlign": "middle",
            },
        )]
    return []

# ---------------------------------------------------------------------------
# STYLE CONSTANTS
# ---------------------------------------------------------------------------

CARD_STYLE = {
    "backgroundColor": DARK_ALT,
    "border": f"1px solid {BORDER_COLOR}",
    "borderRadius": "8px",
    "padding": "20px",
    "marginBottom": "20px",
}

SECTION_HEADER_STYLE = {
    "color": BRAND_ORANGE,
    "fontSize": "0.78rem",
    "fontWeight": "700",
    "letterSpacing": "0.12em",
    "textTransform": "uppercase",
    "marginBottom": "14px",
    "paddingBottom": "8px",
    "borderBottom": f"1px solid {BORDER_COLOR}",
}

LABEL_STYLE = {
    "color": TEXT_SECONDARY,
    "fontSize": "0.75rem",
    "fontWeight": "600",
    "letterSpacing": "0.08em",
    "textTransform": "uppercase",
    "marginBottom": "6px",
    "display": "block",
}

INPUT_STYLE = {
    "backgroundColor": "#1E252D",
    "border": f"1px solid {BORDER_COLOR}",
    "color": TEXT_PRIMARY,
    "borderRadius": "5px",
    "width": "100%",
}

DROPDOWN_STYLE = {"width": "100%"}

# ---------------------------------------------------------------------------
# LAYOUT — INPUT PANEL
# ---------------------------------------------------------------------------

_input_panel = html.Div(
    style={**CARD_STYLE, "borderLeft": f"3px solid {BRAND_ORANGE}"},
    children=[
        html.Div("Match Setup", style=SECTION_HEADER_STYLE),
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr 1fr 1fr",
                "gap": "16px",
                "marginBottom": "16px",
            },
            children=[
                html.Div([
                    html.Label("Our Team", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="prep-our-team",
                        options=[{"label": t, "value": t} for t in PSL_TEAMS],
                        value="Lahore Qalandars",
                        style=DROPDOWN_STYLE,
                        className="psl-dropdown",
                        clearable=False,
                    ),
                ]),
                html.Div([
                    html.Label("Opposition", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="prep-opposition",
                        options=[{"label": t, "value": t} for t in PSL_TEAMS],
                        value="Karachi Kings",
                        style=DROPDOWN_STYLE,
                        className="psl-dropdown",
                        clearable=False,
                    ),
                ]),
                html.Div([
                    html.Label("Venue", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="prep-venue",
                        options=[{"label": v, "value": v} for v in PSL_VENUES],
                        value="Gaddafi Stadium, Lahore",
                        style=DROPDOWN_STYLE,
                        className="psl-dropdown",
                        clearable=False,
                    ),
                ]),
                html.Div([
                    html.Label("Match Date & Time", style=LABEL_STYLE),
                    dcc.DatePickerSingle(
                        id="prep-match-date",
                        date=str(date.today()),
                        display_format="DD MMM YYYY",
                        style={"width": "100%"},
                        className="psl-datepicker",
                    ),
                ]),
            ],
        ),
        # Squad selector
        html.Div([
            html.Label(
                "Our Squad — select 11-18 players",
                style={**LABEL_STYLE, "marginBottom": "8px"},
            ),
            html.Div(
                style={
                    "display": "flex",
                    "gap": "8px",
                    "marginBottom": "10px",
                    "alignItems": "center",
                },
                children=[
                    html.Button(
                        "Select All",
                        id="prep-squad-select-all",
                        n_clicks=0,
                        style={
                            "backgroundColor": "transparent",
                            "border": f"1px solid {BORDER_COLOR}",
                            "color": TEXT_SECONDARY,
                            "padding": "4px 12px",
                            "borderRadius": "4px",
                            "cursor": "pointer",
                            "fontSize": "0.75rem",
                        },
                    ),
                    html.Button(
                        "Clear",
                        id="prep-squad-clear",
                        n_clicks=0,
                        style={
                            "backgroundColor": "transparent",
                            "border": f"1px solid {BORDER_COLOR}",
                            "color": TEXT_SECONDARY,
                            "padding": "4px 12px",
                            "borderRadius": "4px",
                            "cursor": "pointer",
                            "fontSize": "0.75rem",
                        },
                    ),
                    html.Span(
                        id="prep-squad-count",
                        style={"color": TEXT_SECONDARY, "fontSize": "0.75rem"},
                    ),
                ],
            ),
            dcc.Checklist(
                id="prep-squad",
                options=[],
                value=[],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(6, 1fr)",
                    "gap": "4px",
                    "maxHeight": "140px",
                    "overflowY": "auto",
                    "padding": "8px",
                    "backgroundColor": "#1E252D",
                    "borderRadius": "5px",
                    "border": f"1px solid {BORDER_COLOR}",
                },
                inputStyle={"marginRight": "6px", "accentColor": BRAND_ORANGE},
                labelStyle={
                    "color": TEXT_PRIMARY,
                    "fontSize": "0.80rem",
                    "cursor": "pointer",
                },
            ),
        ]),
        # Captain + Must-play row
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 2fr",
                "gap": "16px",
                "marginTop": "16px",
            },
            children=[
                html.Div([
                    html.Label("Captain", style={**LABEL_STYLE, "marginBottom": "6px"}),
                    dcc.Dropdown(
                        id="prep-captain",
                        options=[],
                        value=None,
                        multi=False,
                        placeholder="Select captain...",
                        className="psl-dropdown",
                        style={"backgroundColor": "#1E252D", "border": f"1px solid {BORDER_COLOR}"},
                    ),
                ]),
                html.Div([
                    html.Label(
                        "Must-play players (always included in XI alongside captain)",
                        style={**LABEL_STYLE, "marginBottom": "6px"},
                    ),
                    dcc.Dropdown(
                        id="prep-forced-players",
                        options=[],
                        value=[],
                        multi=True,
                        placeholder="Select key players to lock into XI...",
                        className="psl-dropdown",
                        style={"backgroundColor": "#1E252D", "border": f"1px solid {BORDER_COLOR}"},
                    ),
                ]),
            ],
        ),
        # Generate button
        html.Div(
            style={"marginTop": "20px", "display": "flex", "gap": "12px", "alignItems": "center"},
            children=[
                html.Button(
                    "GENERATE FULL BRIEF",
                    id="prep-generate-btn",
                    n_clicks=0,
                    style={
                        "backgroundColor": BRAND_ORANGE,
                        "color": "#000",
                        "fontWeight": "800",
                        "fontSize": "0.85rem",
                        "letterSpacing": "0.10em",
                        "border": "none",
                        "padding": "12px 28px",
                        "borderRadius": "5px",
                        "cursor": "pointer",
                    },
                ),
                html.Div(id="prep-status", style={"color": TEXT_SECONDARY, "fontSize": "0.80rem"}),
            ],
        ),
    ],
)

# ---------------------------------------------------------------------------
# LAYOUT — OUTPUT SECTIONS (initially hidden)
# ---------------------------------------------------------------------------

_output_sections = html.Div(
    id="prep-output",
    style={"display": "none"},
    children=[
        # SECTION 1: Weather Dashboard
        html.Div(
            id="prep-weather-section",
            style=CARD_STYLE,
        ),
        # SECTION 2: Playing XI
        html.Div(
            id="prep-xi-section",
            style=CARD_STYLE,
        ),
        # SECTION 3: Toss
        html.Div(
            id="prep-toss-section",
            style=CARD_STYLE,
        ),
        # SECTION 4: Opposition Batting Order
        html.Div(
            id="prep-opposition-section",
            style=CARD_STYLE,
        ),
        # SECTION 5: Bowling Plan
        html.Div(
            id="prep-bowling-plan-section",
            style=CARD_STYLE,
        ),
        # SECTION 6: Batting Scenario Cards
        html.Div(
            id="prep-batting-scenarios-section",
            style=CARD_STYLE,
        ),
        # SECTION 7: Matchup Notes
        html.Div(
            id="prep-matchup-section",
            style=CARD_STYLE,
        ),
        # PDF Download
        html.Div(
            style={"textAlign": "center", "paddingBottom": "40px"},
            children=[
                html.Button(
                    "DOWNLOAD PDF BRIEF",
                    id="prep-pdf-btn",
                    n_clicks=0,
                    style={
                        "backgroundColor": "transparent",
                        "border": f"2px solid {BRAND_ORANGE}",
                        "color": BRAND_ORANGE,
                        "fontWeight": "700",
                        "fontSize": "0.85rem",
                        "letterSpacing": "0.10em",
                        "padding": "12px 28px",
                        "borderRadius": "5px",
                        "cursor": "pointer",
                    },
                ),
                dcc.Download(id="prep-pdf-download"),
            ],
        ),
    ],
)

# Store for generated brief data (serialised subset for client)
_store = dcc.Store(id="prep-brief-store", storage_type="memory")

# ---------------------------------------------------------------------------
# PAGE LAYOUT
# ---------------------------------------------------------------------------

layout = html.Div(
    style={"backgroundColor": DARK_BG, "minHeight": "100vh", "padding": "24px"},
    children=[
        _store,
        html.H1(
            "Match Prep Room",
            style={
                "color": TEXT_PRIMARY,
                "fontSize": "1.30rem",
                "fontWeight": "700",
                "letterSpacing": "0.04em",
                "marginBottom": "20px",
            },
        ),
        _input_panel,
        _output_sections,
    ],
)


# ---------------------------------------------------------------------------
# CALLBACK: populate squad checklist when team changes
# ---------------------------------------------------------------------------

@callback(
    Output("prep-squad", "options"),
    Output("prep-squad", "value"),
    Input("prep-our-team", "value"),
)
def _update_squad_options(team: str):
    if not team:
        return [], []
    pi_path = PROJ_ROOT / "data" / "processed" / "player_index_2026_enriched.csv"
    try:
        pi = pd.read_csv(pi_path)
        team_players = pi[pi["current_team_2026"] == team]["player_name"].sort_values().tolist()
    except Exception:
        team_players = []
    dup_names = set()
    try:
        dup_names = set(pi["player_name"][pi["player_name"].duplicated(keep=False)])
    except Exception:
        pass
    options = []
    values = []
    for name in team_players:
        display = f"{name} ({team})" if name in dup_names else name
        options.append({"label": display, "value": display})
        values.append(display)
    return options, values


@callback(
    Output("prep-forced-players", "options"),
    Output("prep-captain",        "options"),
    Input("prep-squad", "value"),
)
def _update_forced_options(squad):
    if not squad:
        return [], []
    opts = [{"label": _strip_disambiguation(p), "value": _strip_disambiguation(p)} for p in squad]
    return opts, opts


@callback(
    Output("prep-squad", "value", allow_duplicate=True),
    Input("prep-squad-select-all", "n_clicks"),
    State("prep-squad", "options"),
    prevent_initial_call=True,
)
def _select_all(_n, options):
    return [o["value"] for o in options]


@callback(
    Output("prep-squad", "value", allow_duplicate=True),
    Input("prep-squad-clear", "n_clicks"),
    prevent_initial_call=True,
)
def _clear_squad(_n):
    return []


@callback(
    Output("prep-squad-count", "children"),
    Input("prep-squad", "value"),
    State("prep-our-team", "value"),
)
def _squad_count(squad, team):
    n = len(squad) if squad else 0
    if team and n > 0:
        return html.Span(
            f"{n} players loaded for {team} \u2014 uncheck unavailable players",
            style={"color": TEXT_SECONDARY},
        )
    colour = GREEN if 11 <= n <= 18 else RED
    return html.Span(f"{n} selected (need 11-18)", style={"color": colour})


# ---------------------------------------------------------------------------
# CALLBACK: generate brief
# ---------------------------------------------------------------------------

@callback(
    Output("prep-output",                  "style"),
    Output("prep-status",                  "children"),
    Output("prep-brief-store",             "data"),
    Output("match-brief-store",            "data"),
    Output("prep-weather-section",         "children"),
    Output("prep-xi-section",              "children"),
    Output("prep-toss-section",            "children"),
    Output("prep-opposition-section",      "children"),
    Output("prep-bowling-plan-section",    "children"),
    Output("prep-batting-scenarios-section","children"),
    Output("prep-matchup-section",         "children"),
    Input("prep-generate-btn", "n_clicks"),
    State("prep-our-team",         "value"),
    State("prep-opposition",       "value"),
    State("prep-venue",            "value"),
    State("prep-match-date",       "date"),
    State("prep-squad",            "value"),
    State("prep-forced-players",   "value"),
    State("prep-captain",          "value"),
    prevent_initial_call=True,
)
def _generate_brief(n_clicks, our_team, opposition, venue, match_date, squad, forced_players, captain):
    if not squad or len(squad) < 11:
        return (
            {"display": "none"},
            html.Span("Select at least 11 squad players.", style={"color": RED}),
            no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update,
        )
    if len(squad) > 18:
        return (
            {"display": "none"},
            html.Span(f"Squad too large ({len(squad)} selected) — maximum is 18. Deselect unavailable players.", style={"color": RED}),
            no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update,
        )
    if our_team == opposition:
        return (
            {"display": "none"},
            html.Span("Our team and opposition cannot be the same.", style={"color": RED}),
            no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update,
        )

    try:
        # Parse date
        dt = datetime.fromisoformat(match_date).replace(hour=19, minute=0)
    except Exception:
        dt = datetime.now().replace(hour=19, minute=0)

    # ----------------------------------------------------------------
    # Generate weather
    # ----------------------------------------------------------------
    try:
        from weather.weather_impact import get_match_weather_impact
        weather = get_match_weather_impact(venue, dt)
    except Exception:
        from utils.situation import WeatherImpact
        weather = WeatherImpact.neutral()

    # ----------------------------------------------------------------
    # Generate brief via decision engine
    # ----------------------------------------------------------------
    clean_squad = [_strip_disambiguation(p) for p in squad]
    clean_forced = [_strip_disambiguation(p) for p in (forced_players or [])]
    # Captain is always locked into the XI — merge with forced (dedup)
    clean_captain = _strip_disambiguation(captain) if captain else None
    if clean_captain and clean_captain not in clean_forced:
        clean_forced = [clean_captain] + clean_forced
    from engine.decision_engine import generate_prematch_brief
    brief = generate_prematch_brief(
        our_team        = our_team,
        opposition      = opposition,
        venue           = venue,
        match_datetime  = dt,
        our_squad       = clean_squad,
        weather_impact  = weather,
        season          = 0,
        innings         = 1,
        forced_players  = clean_forced or None,
        captain         = clean_captain,
    )

    # Serialise key info for PDF callback
    store_data = {
        "our_team":   our_team,
        "opposition": opposition,
        "venue":      venue,
        "match_date": match_date,
        "squad":      squad,
    }

    # ----------------------------------------------------------------
    # Build each section's children
    # ----------------------------------------------------------------
    weather_children   = _render_weather(brief)
    xi_children        = _render_xi(brief)
    toss_children      = _render_toss(brief)
    opposition_children= _render_opposition(brief)
    plan_children      = _render_bowling_plan(brief)
    scenarios_children = _render_batting_scenarios(brief)
    matchup_children   = _render_matchups(brief)

    return (
        {"display": "block"},
        html.Div([
            html.Span(
                f"Brief generated at {datetime.now().strftime('%H:%M:%S')}",
                style={"color": GREEN, "display": "block"},
            ),
        ] + [
            html.Span(
                note,
                style={
                    "color": AMBER if "No PSL" in note else TEXT_SECONDARY,
                    "fontSize": "0.75rem",
                    "display": "block",
                    "marginTop": "2px",
                },
            )
            for note in (brief.data_tier_notes or [])
        ]),
        store_data,
        store_data,   # shared match-brief-store (same payload)
        weather_children,
        xi_children,
        toss_children,
        opposition_children,
        plan_children,
        scenarios_children,
        matchup_children,
    )


# ---------------------------------------------------------------------------
# SECTION RENDERERS
# ---------------------------------------------------------------------------

def _render_weather(brief) -> list:
    w = brief.weather_impact
    warnings = brief.weather_warnings

    def _impact_row(emoji: str, label: str, value: str, colour: str):
        return html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "6px"},
            children=[
                html.Span(emoji, style={"fontSize": "1.1rem"}),
                html.Span(label, style={"color": TEXT_SECONDARY, "fontSize": "0.80rem", "width": "130px"}),
                html.Span(value, style={"color": colour, "fontWeight": "600", "fontSize": "0.82rem"}),
            ],
        )

    spin_pct   = int(w.spinner_penalty * 100)
    swing_pct  = int((w.swing_bonus - 1.0) * 100)
    pace_pct   = int((w.pace_bounce_bonus - 1.0) * 100)
    yorker_pct = int(w.yorker_reliability * 100)

    left_col = html.Div(
        style={"flex": "1"},
        children=[
            _impact_row("🌀", "Spinner effectiveness",
                        f"{spin_pct}%",
                        RED if spin_pct < 70 else AMBER if spin_pct < 85 else GREEN),
            _impact_row("🌬", "Swing bonus",
                        f"+{swing_pct}%" if swing_pct else "Neutral",
                        BRAND_ORANGE if swing_pct >= 20 else GREEN),
            _impact_row("⚡", "Pace / bounce bonus",
                        f"+{pace_pct}%" if pace_pct else "Neutral",
                        BRAND_ORANGE if pace_pct >= 10 else GREEN),
            _impact_row("🎯", "Yorker reliability",
                        f"{yorker_pct}%",
                        RED if yorker_pct < 75 else AMBER if yorker_pct < 88 else GREEN),
            html.Div(
                style={"marginTop": "10px"},
                children=[
                    html.Span(
                        f"Dew onset: over {w.dew_onset_over}" if w.dew_onset_over else "No dew forecast",
                        style={"color": AMBER if w.dew_onset_over else GREEN, "fontSize": "0.82rem"},
                    ),
                ],
            ),
        ],
    )

    right_col = html.Div(
        style={"flex": "1"},
        children=[
            html.Div(
                style={"backgroundColor": "#1E252D", "borderRadius": "6px", "padding": "12px"},
                children=[
                    html.Div(
                        "Cricket Impact",
                        style={**LABEL_STYLE, "marginBottom": "10px"},
                    ),
                ] + [
                    html.Div(
                        style={
                            "display": "flex",
                            "alignItems": "flex-start",
                            "gap": "8px",
                            "marginBottom": "8px",
                            "padding": "8px",
                            "backgroundColor": DARK_ALT,
                            "borderRadius": "5px",
                            "borderLeft": f"3px solid {RED if 'dew' in wt.lower() or 'rain' in wt.lower() else AMBER}",
                        } if wt else {},
                        children=[
                            html.Span(
                                "🔴" if any(k in wt.lower() for k in ("heavy","severe","critical"))
                                else "🟡" if any(k in wt.lower() for k in ("dew","rain","humid"))
                                else "🟢",
                                style={"fontSize": "0.85rem"},
                            ),
                            html.Span(wt, style={"color": TEXT_PRIMARY, "fontSize": "0.80rem", "flex": "1"}),
                        ],
                    )
                    for wt in (warnings if warnings else ["No significant weather concerns."])
                ],
            ),
        ],
    )

    if w.dl_planning_needed:
        right_col.children.append(
            html.Div(
                "D/L planning needed — front-load batting: D/L rewards runs scored early.",
                style={
                    "marginTop": "8px",
                    "backgroundColor": "#2A1A00",
                    "border": f"1px solid {AMBER}",
                    "borderRadius": "5px",
                    "padding": "8px",
                    "color": AMBER,
                    "fontSize": "0.80rem",
                    "fontWeight": "600",
                },
            )
        )

    return [
        html.Div("Weather Dashboard", style=SECTION_HEADER_STYLE),
        html.Div(
            style={"display": "flex", "gap": "24px"},
            children=[left_col, right_col],
        ),
    ]


def _render_xi(brief) -> list:
    try:
        _pi_xi = pd.read_csv(PROJ_ROOT / "data" / "processed" / "player_index_2026_enriched.csv")
    except Exception:
        _pi_xi = pd.DataFrame()
    captain_name = getattr(brief, "captain", None) or ""
    tabs = []
    for opt in brief.xi_options:
        rows = []
        for p in opt.players:
            role_color = {
                "Bowler": BRAND_ORANGE,
                "All-rounder": AMBER,
                "Wicketkeeper": STEEL_BLUE if hasattr(p, "role") else TEXT_SECONDARY,
                "WK-Batsman": "#4A90D9",
            }.get(p.role, GREEN)

            is_captain = (captain_name and p.player_name == captain_name)
            captain_badge = [html.Span(
                "C",
                style={
                    "color": "#000",
                    "backgroundColor": BRAND_ORANGE,
                    "fontSize": "0.60rem",
                    "fontWeight": "900",
                    "padding": "1px 5px",
                    "borderRadius": "3px",
                    "marginLeft": "6px",
                    "verticalAlign": "middle",
                    "letterSpacing": "0.04em",
                },
            )] if is_captain else []

            rows.append(
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "30px 1fr 110px 1fr",
                        "gap": "8px",
                        "alignItems": "center",
                        "padding": "7px 10px",
                        "borderRadius": "4px",
                        "marginBottom": "3px",
                        "backgroundColor": "#1A1F26" if p.batting_position % 2 == 0 else DARK_ALT,
                        "border": f"1px solid {BRAND_ORANGE}40" if is_captain else "1px solid transparent",
                    },
                    children=[
                        html.Span(
                            str(p.batting_position),
                            style={"color": TEXT_SECONDARY, "fontSize": "0.75rem", "textAlign": "center"},
                        ),
                        html.Div(
                            style={"display": "flex", "alignItems": "center"},
                            children=[
                                html.Span(p.player_name, style={"color": TEXT_PRIMARY, "fontWeight": "600", "fontSize": "0.85rem"}),
                                *captain_badge,
                                *_get_tier_badge(p.player_name, _pi_xi),
                            ],
                        ),
                        html.Span(
                            p.role,
                            style={
                                "color": role_color,
                                "fontSize": "0.72rem",
                                "fontWeight": "700",
                                "padding": "2px 7px",
                                "borderRadius": "3px",
                                "backgroundColor": f"{role_color}20",
                                "textAlign": "center",
                                "whiteSpace": "nowrap",
                            },
                        ),
                        html.Span(
                            p.key_stat,
                            style={"color": TEXT_SECONDARY, "fontSize": "0.75rem"},
                        ),
                    ],
                )
            )

        tabs.append(
            dcc.Tab(
                label=f"{opt.label}: {opt.description}",
                value=opt.label,
                style={
                    "backgroundColor": DARK_ALT,
                    "color": TEXT_SECONDARY,
                    "border": f"1px solid {BORDER_COLOR}",
                    "padding": "8px 16px",
                    "fontSize": "0.80rem",
                    "fontWeight": "600",
                },
                selected_style={
                    "backgroundColor": BRAND_ORANGE,
                    "color": "#000",
                    "border": f"1px solid {BRAND_ORANGE}",
                    "padding": "8px 16px",
                    "fontSize": "0.80rem",
                    "fontWeight": "700",
                },
                children=html.Div(
                    style={"paddingTop": "12px"},
                    children=rows + [
                        html.Div(
                            opt.constraint_note,
                            style={
                                "marginTop": "10px",
                                "padding": "8px 12px",
                                "backgroundColor": "#1E252D",
                                "borderRadius": "4px",
                                "color": TEXT_SECONDARY,
                                "fontSize": "0.75rem",
                                "fontWeight": "600",
                            },
                        ),
                        html.Div(
                            f"Total squad score: {opt.total_score:.0f}",
                            style={"color": BRAND_ORANGE, "fontSize": "0.75rem", "marginTop": "6px"},
                        ),
                    ],
                ),
            )
        )

    return [
        html.Div("Playing 11 Recommendation", style=SECTION_HEADER_STYLE),
        dcc.Tabs(
            id="xi-tabs",
            value="Option A",
            children=tabs,
            style={"marginBottom": "4px"},
        ),
    ]


def _render_toss(brief) -> list:
    t = brief.toss
    rec_color = {
        "BAT FIRST":  GREEN,
        "BOWL FIRST": STEEL_BLUE if hasattr(brief, "opposition") else BRAND_ORANGE,
        "NEUTRAL":    AMBER,
    }.get(t.recommendation, TEXT_PRIMARY)

    return [
        html.Div("Toss Recommendation", style=SECTION_HEADER_STYLE),
        html.Div(
            style={"display": "flex", "gap": "24px", "alignItems": "flex-start"},
            children=[
                html.Div(
                    style={
                        "backgroundColor": f"{rec_color}15",
                        "border": f"2px solid {rec_color}",
                        "borderRadius": "8px",
                        "padding": "20px 30px",
                        "textAlign": "center",
                        "minWidth": "200px",
                    },
                    children=[
                        html.Div(
                            t.recommendation,
                            style={
                                "color": rec_color,
                                "fontSize": "1.6rem",
                                "fontWeight": "900",
                                "letterSpacing": "0.06em",
                                "marginBottom": "4px",
                            },
                        ),
                        html.Div(
                            brief.venue.split(",")[0],
                            style={"color": TEXT_SECONDARY, "fontSize": "0.75rem"},
                        ),
                    ],
                ),
                html.Div(
                    style={"flex": "1"},
                    children=[
                        html.Div(
                            "Reasoning:",
                            style={"color": TEXT_SECONDARY, "fontSize": "0.75rem", "marginBottom": "8px"},
                        ),
                    ] + [
                        html.Div(
                            style={
                                "display": "flex",
                                "gap": "8px",
                                "marginBottom": "8px",
                                "alignItems": "flex-start",
                            },
                            children=[
                                html.Span("•", style={"color": BRAND_ORANGE, "fontWeight": "700", "marginTop": "1px"}),
                                html.Span(r, style={"color": TEXT_PRIMARY, "fontSize": "0.83rem"}),
                            ],
                        )
                        for r in t.reasoning
                    ] + ([
                        html.Div(
                            style={
                                "marginTop": "10px",
                                "padding": "8px 12px",
                                "backgroundColor": "#2A1A00",
                                "border": f"1px solid {AMBER}",
                                "borderRadius": "5px",
                            },
                            children=[
                                html.Span("D/L Note: ", style={"color": AMBER, "fontWeight": "700", "fontSize": "0.80rem"}),
                                html.Span(t.dl_note, style={"color": TEXT_PRIMARY, "fontSize": "0.80rem"}),
                            ],
                        )
                    ] if t.dl_note else []),
                ),
            ],
        ),
    ]


def _render_opposition(brief) -> list:
    opp = brief.opposition_order
    rows = []

    for pb in opp.predicted_order:
        danger_col = DANGER_COLORS.get(pb.danger_rating, TEXT_SECONDARY)
        conf_col   = CONFIDENCE_COLORS.get(pb.confidence, TEXT_SECONDARY)

        rows.append(
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "30px 1fr 80px 100px 90px 1fr",
                    "gap": "8px",
                    "alignItems": "center",
                    "padding": "8px 10px",
                    "borderRadius": "4px",
                    "marginBottom": "3px",
                    "backgroundColor": "#1A1F26" if pb.position % 2 == 0 else DARK_ALT,
                    "borderLeft": f"3px solid {danger_col}",
                },
                children=[
                    html.Span(str(pb.position), style={"color": TEXT_SECONDARY, "fontSize": "0.75rem", "textAlign": "center"}),
                    html.Span(pb.player_name, style={"color": TEXT_PRIMARY, "fontWeight": "600", "fontSize": "0.85rem"}),
                    html.Span(
                        pb.danger_rating,
                        style={
                            "color": danger_col,
                            "fontSize": "0.70rem",
                            "fontWeight": "700",
                            "padding": "2px 6px",
                            "borderRadius": "3px",
                            "backgroundColor": f"{danger_col}20",
                            "textAlign": "center",
                        },
                    ),
                    html.Span(
                        pb.arrival_over_range,
                        style={"color": TEXT_SECONDARY, "fontSize": "0.75rem"},
                    ),
                    html.Span(
                        pb.confidence,
                        style={"color": conf_col, "fontSize": "0.70rem"},
                    ),
                    html.Span(
                        pb.key_note,
                        style={"color": TEXT_SECONDARY, "fontSize": "0.75rem"},
                    ),
                ],
            )
        )

    # Column headers
    header = html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "30px 1fr 80px 100px 90px 1fr",
            "gap": "8px",
            "padding": "6px 10px",
            "marginBottom": "6px",
        },
        children=[
            html.Span(h, style={"color": TEXT_SECONDARY, "fontSize": "0.70rem", "fontWeight": "700", "letterSpacing": "0.08em"})
            for h in ["#", "PLAYER", "DANGER", "ARRIVAL", "CONF.", "KEY NOTE"]
        ],
    )

    implications = html.Div(
        style={"marginTop": "14px", "padding": "12px", "backgroundColor": "#1E252D", "borderRadius": "6px"},
        children=[
            html.Div(
                "Bowling Implications",
                style={**LABEL_STYLE, "marginBottom": "10px"},
            ),
        ] + [
            html.Div(
                style={"display": "flex", "gap": "8px", "marginBottom": "6px"},
                children=[
                    html.Span("→", style={"color": BRAND_ORANGE, "fontWeight": "700"}),
                    html.Span(note, style={"color": TEXT_PRIMARY, "fontSize": "0.82rem"}),
                ],
            )
            for note in opp.bowling_implications
        ],
    )

    _NEW_FRANCHISES = {"Pindiz", "Hyderabad Houston Kings"}
    estimated_banner = []
    if getattr(opp, "is_estimated", False) or brief.opposition in _NEW_FRANCHISES:
        estimated_banner = [html.Div(
            style={
                "display": "flex", "gap": "10px", "alignItems": "flex-start",
                "padding": "10px 14px", "marginBottom": "12px",
                "backgroundColor": "#2A1800",
                "border": f"1px solid {AMBER}",
                "borderRadius": "6px",
            },
            children=[
                html.Span("\u26a0", style={"color": AMBER, "fontSize": "1rem"}),
                html.Div([
                    html.Span(
                        "New franchise \u2014 no PSL match history available. "
                        "Opposition analysis is based on squad composition only.",
                        style={"color": AMBER, "fontWeight": "700", "fontSize": "0.82rem"},
                    ),
                ]),
            ],
        )]

    return estimated_banner + [
        html.Div(
            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
            children=[
                html.Div(f"Opposition Batting Order — {brief.opposition}", style=SECTION_HEADER_STYLE),
                html.Span(
                    f"Season: {opp.season}  |  {opp.team}",
                    style={"color": TEXT_SECONDARY, "fontSize": "0.72rem"},
                ),
            ],
        ),
        header,
    ] + rows + [implications]


def _render_bowling_plan(brief) -> list:
    try:
        _pi_bp = pd.read_csv(PROJ_ROOT / "data" / "processed" / "player_index_2026_enriched.csv")
    except Exception:
        _pi_bp = pd.DataFrame()
    plan = brief.bowling_plan
    overs = plan.overs

    # Phase colour
    def _phase_color(phase: str) -> str:
        return {
            "Powerplay": "#4A90D9",
            "Middle": AMBER,
            "Death": RED,
        }.get(phase, TEXT_SECONDARY)

    # Build 20 over boxes in a 4-row x 5-col grid (overs 1-5, 6-10, 11-15, 16-20)
    boxes = []
    for oa in overs:
        phase_col = _phase_color(oa.phase)
        has_warning = bool(oa.weather_note)
        boxes.append(
            html.Div(
                style={
                    "backgroundColor": "#1A1F26",
                    "border": f"1px solid {AMBER if has_warning else BORDER_COLOR}",
                    "borderTop": f"3px solid {phase_col}",
                    "borderRadius": "5px",
                    "padding": "8px",
                    "minHeight": "72px",
                    "overflow": "visible",
                },
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"},
                        children=[
                            html.Span(
                                f"Over {oa.over}",
                                style={"color": TEXT_SECONDARY, "fontSize": "0.68rem", "fontWeight": "700"},
                            ),
                            html.Span(
                                oa.phase[:2],
                                style={"color": phase_col, "fontSize": "0.65rem", "fontWeight": "700"},
                            ),
                        ],
                    ),
                    html.Div(
                        oa.primary_bowler.split()[-1],
                        style={"color": TEXT_PRIMARY, "fontWeight": "700", "fontSize": "0.82rem", "marginBottom": "2px"},
                    ),
                    html.Div(
                        oa.backup_bowler.split()[-1] if oa.backup_bowler else "",
                        style={"color": TEXT_SECONDARY, "fontSize": "0.70rem"},
                    ),
                    html.Div(
                        "⚠ " + oa.weather_note if has_warning else "",
                        style={"color": AMBER, "fontSize": "0.65rem", "marginTop": "2px", "whiteSpace": "normal"},
                    ),
                ],
            )
        )

    grid = html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(10, 1fr)",
            "gap": "6px",
            "marginBottom": "16px",
        },
        children=boxes,
    )

    # Bowler summary
    summary_items = []
    for bowler, allocated in plan.bowler_summary.items():
        summary_items.append(
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "8px",
                    "marginBottom": "4px",
                    "padding": "6px 10px",
                    "backgroundColor": "#1E252D",
                    "borderRadius": "4px",
                },
                children=[
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "minWidth": "180px"},
                        children=[
                            html.Span(bowler, style={"color": TEXT_PRIMARY, "fontWeight": "600", "fontSize": "0.82rem"}),
                            *_get_tier_badge(bowler, _pi_bp),
                        ],
                    ),
                    html.Div(
                        style={"display": "flex", "gap": "3px"},
                        children=[
                            html.Span(
                                str(o),
                                style={
                                    "color": "#000",
                                    "backgroundColor": BRAND_ORANGE,
                                    "fontSize": "0.70rem",
                                    "fontWeight": "700",
                                    "padding": "2px 6px",
                                    "borderRadius": "3px",
                                    "minWidth": "22px",
                                    "textAlign": "center",
                                },
                            )
                            for o in allocated
                        ],
                    ),
                    html.Span(
                        f"{len(allocated)} overs",
                        style={"color": TEXT_SECONDARY, "fontSize": "0.72rem", "marginLeft": "4px"},
                    ),
                ],
            )
        )

    # Contingency notes
    contingency = []
    for note in plan.contingencies:
        contingency.append(
            html.Div(
                style={"display": "flex", "gap": "8px", "marginBottom": "5px"},
                children=[
                    html.Span("⚡", style={"fontSize": "0.80rem"}),
                    html.Span(note, style={"color": TEXT_SECONDARY, "fontSize": "0.80rem"}),
                ],
            )
        )

    return [
        html.Div("Over-by-Over Bowling Plan", style=SECTION_HEADER_STYLE),
        # Phase legend
        html.Div(
            style={"display": "flex", "gap": "16px", "marginBottom": "10px"},
            children=[
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "5px"}, children=[
                    html.Div(style={"width": "12px", "height": "12px", "backgroundColor": "#4A90D9", "borderRadius": "2px"}),
                    html.Span("Powerplay", style={"color": TEXT_SECONDARY, "fontSize": "0.72rem"}),
                ]),
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "5px"}, children=[
                    html.Div(style={"width": "12px", "height": "12px", "backgroundColor": AMBER, "borderRadius": "2px"}),
                    html.Span("Middle", style={"color": TEXT_SECONDARY, "fontSize": "0.72rem"}),
                ]),
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "5px"}, children=[
                    html.Div(style={"width": "12px", "height": "12px", "backgroundColor": RED, "borderRadius": "2px"}),
                    html.Span("Death", style={"color": TEXT_SECONDARY, "fontSize": "0.72rem"}),
                ]),
                html.Span("Primary / backup listed", style={"color": TEXT_SECONDARY, "fontSize": "0.72rem", "marginLeft": "8px"}),
            ],
        ),
        grid,
        html.Div("Bowler Allocation", style={**LABEL_STYLE, "marginTop": "8px", "marginBottom": "8px"}),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "6px", "marginBottom": "14px"},
            children=summary_items,
        ),
        html.Div("Contingency Notes", style={**LABEL_STYLE, "marginBottom": "8px"}),
        html.Div(children=contingency),
    ]


def _render_batting_scenarios(brief) -> list:
    cards = brief.batting_scenarios
    scenario_colors = {
        "A": GREEN,
        "B": RED,
        "C": BRAND_ORANGE,
        "D": AMBER,
    }

    card_divs = []
    for sc in cards:
        col = scenario_colors.get(sc.scenario_id, TEXT_SECONDARY)

        batter_rows = []
        for bi in sc.batting_order[:8]:
            role_col = {
                "Anchor":     AMBER,
                "Aggressor":  GREEN,
                "Finisher":   BRAND_ORANGE,
                "Stabiliser": "#4A90D9",
                "Support":    TEXT_SECONDARY,
            }.get(bi.role_in_card, TEXT_SECONDARY)

            batter_rows.append(
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "20px 1fr 80px",
                        "gap": "6px",
                        "alignItems": "flex-start",
                        "marginBottom": "5px",
                        "paddingBottom": "5px",
                        "borderBottom": f"1px solid {BORDER_COLOR}",
                    },
                    children=[
                        html.Span(str(bi.position), style={"color": TEXT_SECONDARY, "fontSize": "0.72rem"}),
                        html.Div([
                            html.Span(bi.player_name, style={"color": TEXT_PRIMARY, "fontWeight": "600", "fontSize": "0.80rem", "display": "block"}),
                            html.Span(bi.instruction, style={"color": TEXT_SECONDARY, "fontSize": "0.72rem"}),
                        ]),
                        html.Span(
                            bi.role_in_card,
                            style={
                                "color": role_col,
                                "fontSize": "0.65rem",
                                "fontWeight": "700",
                                "padding": "2px 5px",
                                "borderRadius": "3px",
                                "backgroundColor": f"{role_col}20",
                                "textAlign": "center",
                                "whiteSpace": "nowrap",
                            },
                        ),
                    ],
                )
            )

        card_divs.append(
            html.Div(
                style={
                    "backgroundColor": "#1A1F26",
                    "border": f"1px solid {col}40",
                    "borderTop": f"3px solid {col}",
                    "borderRadius": "6px",
                    "padding": "14px",
                    "flex": "1",
                    "minWidth": "0",
                },
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"},
                        children=[
                            html.Span(
                                sc.name,
                                style={"color": col, "fontWeight": "700", "fontSize": "0.85rem"},
                            ),
                            html.Span(
                                f"[{sc.scenario_id}]",
                                style={"color": col, "fontWeight": "900", "fontSize": "0.80rem"},
                            ),
                        ],
                    ),
                    html.Div(
                        sc.key_message,
                        style={
                            "color": TEXT_PRIMARY,
                            "fontSize": "0.78rem",
                            "fontStyle": "italic",
                            "marginBottom": "8px",
                            "paddingBottom": "8px",
                            "borderBottom": f"1px solid {BORDER_COLOR}",
                        },
                    ),
                    html.Div(
                        style={"marginBottom": "6px"},
                        children=[
                            html.Span("Trigger: ", style={"color": TEXT_SECONDARY, "fontSize": "0.70rem", "fontWeight": "700"}),
                            html.Span(sc.trigger, style={"color": TEXT_SECONDARY, "fontSize": "0.70rem"}),
                        ],
                    ),
                    html.Div(children=batter_rows),
                ] + ([
                    html.Div(
                        style={
                            "marginTop": "8px",
                            "padding": "6px 10px",
                            "backgroundColor": "#2A1800",
                            "borderRadius": "4px",
                            "borderLeft": f"3px solid {AMBER}",
                        },
                        children=[
                            html.Span("Weather: ", style={"color": AMBER, "fontWeight": "700", "fontSize": "0.72rem"}),
                            html.Span(sc.weather_note, style={"color": TEXT_SECONDARY, "fontSize": "0.72rem"}),
                        ],
                    )
                ] if sc.weather_note else []),
            )
        )

    return [
        html.Div("Batting Scenario Cards", style=SECTION_HEADER_STYLE),
        html.Div(
            style={"display": "flex", "gap": "12px"},
            children=card_divs,
        ),
    ]


def _render_matchups(brief) -> list:
    notes = brief.matchup_notes

    if not notes:
        return [
            html.Div("Key Matchup Notes", style=SECTION_HEADER_STYLE),
            html.Div(
                "Insufficient H2H data — no matchup notes with 8+ balls in PSL history.",
                style={"color": TEXT_SECONDARY, "fontSize": "0.82rem", "fontStyle": "italic"},
            ),
        ]

    conf_order = {"High": 0, "Medium": 1, "Low": 2, "Insufficient": 3}
    sorted_notes = sorted(notes, key=lambda n: conf_order.get(n.confidence, 9))

    rows = []
    for mn in sorted_notes:
        conf_col = CONFIDENCE_COLORS.get(mn.confidence, TEXT_SECONDARY)
        rows.append(
            html.Div(
                style={
                    "display": "flex",
                    "gap": "12px",
                    "alignItems": "flex-start",
                    "padding": "10px 14px",
                    "marginBottom": "6px",
                    "backgroundColor": "#1A1F26",
                    "borderRadius": "5px",
                    "borderLeft": f"3px solid {conf_col}",
                },
                children=[
                    html.Div(
                        style={"minWidth": "70px"},
                        children=[
                            html.Span(
                                mn.confidence,
                                style={
                                    "color": conf_col,
                                    "fontSize": "0.68rem",
                                    "fontWeight": "700",
                                    "padding": "2px 6px",
                                    "borderRadius": "3px",
                                    "backgroundColor": f"{conf_col}20",
                                },
                            ),
                            html.Div(
                                f"{mn.balls} balls",
                                style={"color": TEXT_SECONDARY, "fontSize": "0.65rem", "marginTop": "3px"},
                            ),
                        ],
                    ),
                    html.Span(
                        mn.note,
                        style={"color": TEXT_PRIMARY, "fontSize": "0.83rem", "flex": "1"},
                    ),
                ],
            )
        )

    return [
        html.Div(
            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
            children=[
                html.Div("Key Matchup Notes", style=SECTION_HEADER_STYLE),
                html.Span(
                    f"{len(notes)} matchup(s) with meaningful H2H data",
                    style={"color": TEXT_SECONDARY, "fontSize": "0.72rem"},
                ),
            ],
        ),
    ] + rows


# ---------------------------------------------------------------------------
# CALLBACK: PDF download
# ---------------------------------------------------------------------------

@callback(
    Output("prep-pdf-download", "data"),
    Input("prep-pdf-btn", "n_clicks"),
    State("prep-brief-store", "data"),
    State("prep-our-team",   "value"),
    State("prep-opposition", "value"),
    State("prep-venue",      "value"),
    State("prep-match-date", "date"),
    State("prep-squad",      "value"),
    prevent_initial_call=True,
)
def _download_pdf(n_clicks, store_data, our_team, opposition, venue, match_date, squad):
    if not squad or len(squad) < 11:
        return no_update

    try:
        dt = datetime.fromisoformat(match_date).replace(hour=19, minute=0)
    except Exception:
        dt = datetime.now().replace(hour=19, minute=0)

    try:
        from weather.weather_impact import get_match_weather_impact
        weather = get_match_weather_impact(venue, dt)
    except Exception:
        from utils.situation import WeatherImpact
        weather = WeatherImpact.neutral()

    clean_squad_pdf = [_strip_disambiguation(p) for p in squad]
    from engine.decision_engine import generate_prematch_brief
    brief = generate_prematch_brief(
        our_team       = our_team,
        opposition     = opposition,
        venue          = venue,
        match_datetime = dt,
        our_squad      = clean_squad_pdf,
        weather_impact = weather,
        season         = 0,
        innings        = 1,
    )

    from utils.pdf_generator import generate_pdf
    out_path = generate_pdf(brief)

    with open(out_path, "rb") as f:
        pdf_bytes = f.read()

    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    filename = f"PSL_Brief_{our_team.replace(' ','_')}_vs_{opposition.replace(' ','_')}_{dt.strftime('%Y%m%d')}.pdf"

    return dcc.send_bytes(pdf_bytes, filename)
