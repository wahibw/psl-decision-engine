# pages/player_profiles.py
# PAGE 3 — Player Profiles
# Search any PSL player — batting + bowling stats, phase breakdown,
# recent form, matchup panel (victims / dismissers)
# Route: /players

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import dash
from dash import html, dcc, Input, Output, State, callback, no_update

from utils.theme import (
    BRAND_ORANGE, DARK_BG, DARK_ALT, TEXT_PRIMARY, TEXT_SECONDARY,
    BORDER_COLOR, GREEN, AMBER, RED, STEEL_BLUE,
    DANGER_COLORS,
)
# Importing theme registers "psl_dark" as the default plotly template
import utils.theme as _theme  # noqa: F401
PSL_TEMPLATE = "psl_dark"

dash.register_page(__name__, path="/players", name="Players")

PROJ_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# STYLE CONSTANTS
# ---------------------------------------------------------------------------

CARD = {
    "backgroundColor": DARK_ALT,
    "border": f"1px solid {BORDER_COLOR}",
    "borderRadius": "8px",
    "padding": "18px",
    "marginBottom": "16px",
}

SECTION_HDR = {
    "color": BRAND_ORANGE,
    "fontSize": "0.75rem",
    "fontWeight": "700",
    "letterSpacing": "0.12em",
    "textTransform": "uppercase",
    "marginBottom": "12px",
    "paddingBottom": "6px",
    "borderBottom": f"1px solid {BORDER_COLOR}",
}

STAT_VAL = {
    "color": TEXT_PRIMARY,
    "fontSize": "1.20rem",
    "fontWeight": "800",
    "lineHeight": "1.1",
}

STAT_LBL = {
    "color": TEXT_SECONDARY,
    "fontSize": "0.62rem",
    "letterSpacing": "0.10em",
    "textTransform": "uppercase",
}

# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_player_stats() -> pd.DataFrame:
    path = PROJ_ROOT / "data" / "processed" / "player_stats.parquet"
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def _load_matchup_matrix() -> pd.DataFrame:
    path = PROJ_ROOT / "data" / "processed" / "matchup_matrix.parquet"
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def _load_ball_by_ball() -> pd.DataFrame:
    path = PROJ_ROOT / "data" / "processed" / "ball_by_ball.parquet"
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def _load_player_index() -> pd.DataFrame:
    path = PROJ_ROOT / "data" / "processed" / "player_index_2026_enriched.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["player_name", "primary_role", "batting_style", "bowling_style", "is_overseas"])


@lru_cache(maxsize=1)
def _all_players() -> list[str]:
    ps = _load_player_stats()
    return sorted(ps["player_name"].unique().tolist())


# ---------------------------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------------------------

layout = html.Div(
    style={"backgroundColor": DARK_BG, "minHeight": "100vh", "padding": "24px"},
    children=[
        html.H1(
            "Player Profiles",
            style={"color": TEXT_PRIMARY, "fontSize": "1.30rem", "fontWeight": "700",
                   "letterSpacing": "0.04em", "marginBottom": "20px"},
        ),
        # Search bar
        html.Div(
            style={
                **CARD,
                "borderLeft": f"3px solid {BRAND_ORANGE}",
                "display": "flex",
                "gap": "12px",
                "alignItems": "center",
                "padding": "14px 20px",
            },
            children=[
                html.Span("SEARCH PLAYER", style={
                    "color": TEXT_SECONDARY,
                    "fontSize": "0.72rem",
                    "fontWeight": "700",
                    "letterSpacing": "0.10em",
                    "whiteSpace": "nowrap",
                }),
                dcc.Dropdown(
                    id="pp-player-select",
                    options=[{"label": p, "value": p} for p in _all_players()],
                    placeholder="Type a player name…",
                    className="psl-dropdown",
                    style={"flex": "1"},
                    clearable=True,
                ),
                html.Span(
                    id="pp-player-meta",
                    style={"color": TEXT_SECONDARY, "fontSize": "0.78rem", "whiteSpace": "nowrap"},
                ),
            ],
        ),
        # Profile output
        html.Div(id="pp-profile-output"),
    ],
)


# ---------------------------------------------------------------------------
# CALLBACK
# ---------------------------------------------------------------------------

@callback(
    Output("pp-profile-output",  "children"),
    Output("pp-player-meta",     "children"),
    Input("pp-player-select",    "value"),
)
def _render_profile(player: str | None):
    if not player:
        return (
            html.Div(
                "Select a player above to view their PSL career profile.",
                style={"color": TEXT_SECONDARY, "padding": "40px", "textAlign": "center",
                       "fontSize": "0.90rem", "fontStyle": "italic"},
            ),
            "",
        )

    ps  = _load_player_stats()
    mm  = _load_matchup_matrix()
    pi  = _load_player_index()
    bb  = _load_ball_by_ball()

    # Meta from player index
    meta_row = pi[pi["player_name"] == player]
    role     = meta_row["primary_role"].iloc[0]   if not meta_row.empty else "—"
    bat_st   = meta_row["batting_style"].iloc[0]  if not meta_row.empty else "—"
    bowl_st  = meta_row["bowling_style"].iloc[0]  if not meta_row.empty else "—"
    overseas = meta_row["is_overseas"].iloc[0]    if not meta_row.empty else False

    meta_label = f"{role}  ·  {bat_st}  ·  {bowl_st}  ·  {'Overseas' if overseas else 'Local'}"

    # Career overall stats
    career = ps[(ps["player_name"] == player) & (ps["season"] == 0) & (ps["phase"] == "overall")]
    career = career.iloc[0] if not career.empty else None

    # Phase breakdown (career)
    phases = ps[(ps["player_name"] == player) & (ps["season"] == 0) & (ps["phase"] != "overall")]

    sections = []

    # Header card
    sections.append(_render_header_card(player, role, bat_st, bowl_st, overseas, career))

    # Batting section
    bat_rows = phases[phases["bat_innings"] > 0].sort_values("bat_innings", ascending=False)
    if career is not None and (career["bat_innings"] or 0) > 0:
        sections.append(_render_batting_section(career, bat_rows))

    # Bowling section
    if career is not None and not pd.isna(career.get("bowl_economy", float("nan"))):
        sections.append(_render_bowling_section(career, phases))

    # Recent form (last 8 innings/spells from ball-by-ball)
    recent_sec = _render_recent_form(player, role, bb)
    if recent_sec:
        sections.append(recent_sec)

    # Matchup panel
    sections.append(_render_matchup_panel(player, role, mm))

    return html.Div(sections), meta_label


# ---------------------------------------------------------------------------
# SECTION RENDERERS
# ---------------------------------------------------------------------------

def _stat_box(value: str, label: str, colour: str = TEXT_PRIMARY) -> html.Div:
    return html.Div(
        style={"textAlign": "center", "padding": "10px 14px",
               "backgroundColor": "#1E252D", "borderRadius": "5px", "minWidth": "80px"},
        children=[
            html.Div(value, style={**STAT_VAL, "color": colour}),
            html.Div(label, style=STAT_LBL),
        ],
    )


def _render_header_card(player, role, bat_st, bowl_st, overseas, career) -> html.Div:
    inns = int(career["bat_innings"]) if career is not None and not pd.isna(career["bat_innings"]) else 0
    runs = int(career["bat_runs"])    if career is not None and not pd.isna(career["bat_runs"])    else 0

    role_col = {
        "Batsman": GREEN, "WK-Batsman": STEEL_BLUE, "Wicketkeeper": STEEL_BLUE,
        "All-rounder": AMBER, "Bowler": BRAND_ORANGE,
    }.get(role, TEXT_SECONDARY)

    return html.Div(
        style={
            **CARD,
            "display": "flex",
            "alignItems": "center",
            "gap": "20px",
            "borderLeft": f"4px solid {role_col}",
        },
        children=[
            html.Div(
                player[0].upper(),
                style={
                    "width": "64px", "height": "64px",
                    "backgroundColor": f"{role_col}25",
                    "border": f"2px solid {role_col}",
                    "borderRadius": "50%",
                    "display": "flex", "alignItems": "center", "justifyContent": "center",
                    "fontSize": "1.60rem", "fontWeight": "900", "color": role_col,
                    "flexShrink": "0",
                },
            ),
            html.Div([
                html.Div(player, style={"color": TEXT_PRIMARY, "fontSize": "1.40rem", "fontWeight": "800"}),
                html.Div(
                    style={"display": "flex", "gap": "8px", "marginTop": "4px", "flexWrap": "wrap"},
                    children=[
                        html.Span(role, style={
                            "color": role_col, "fontSize": "0.72rem", "fontWeight": "700",
                            "padding": "2px 8px", "borderRadius": "3px", "backgroundColor": f"{role_col}20",
                        }),
                        html.Span(bat_st, style={"color": TEXT_SECONDARY, "fontSize": "0.72rem"}),
                        html.Span("·", style={"color": BORDER_COLOR}),
                        html.Span(bowl_st if bowl_st and bowl_st != "nan" else "No bowling", style={"color": TEXT_SECONDARY, "fontSize": "0.72rem"}),
                        html.Span("·", style={"color": BORDER_COLOR}),
                        html.Span(
                            "OVERSEAS" if overseas else "LOCAL",
                            style={"color": AMBER if overseas else GREEN, "fontSize": "0.70rem", "fontWeight": "700"},
                        ),
                    ],
                ),
                html.Div(
                    f"PSL Career: {inns} innings  ·  {runs} runs",
                    style={"color": TEXT_SECONDARY, "fontSize": "0.75rem", "marginTop": "4px"},
                ),
            ]),
        ],
    )


def _render_batting_section(career, phases: pd.DataFrame) -> html.Div:
    def _safe(row, col, fmt=".1f", fallback="—"):
        v = row.get(col, None) if isinstance(row, dict) else getattr(row, col, None)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return fallback
        if fmt:
            return f"{v:{fmt}}"
        return str(int(v))

    avg_col = GREEN if float(career.get("bat_avg", 0) or 0) >= 35 else AMBER if float(career.get("bat_avg", 0) or 0) >= 20 else RED
    sr_col  = GREEN if float(career.get("bat_sr",  0) or 0) >= 140 else AMBER if float(career.get("bat_sr",  0) or 0) >= 110 else RED

    # Stats row
    stats_row = html.Div(
        style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "16px"},
        children=[
            _stat_box(_safe(career, "bat_innings", fmt="", fallback="0"), "INNINGS"),
            _stat_box(_safe(career, "bat_runs", fmt="", fallback="0"), "RUNS"),
            _stat_box(_safe(career, "bat_avg"), "AVERAGE", avg_col),
            _stat_box(_safe(career, "bat_sr"), "STRIKE RATE", sr_col),
            _stat_box(_safe(career, "bat_boundary_pct"), "BOUNDARY %"),
            _stat_box(_safe(career, "bat_dot_pct"), "DOT %"),
        ],
    )

    # Phase breakdown bar chart
    phase_order = ["powerplay", "middle", "death"]
    phase_labels = {"powerplay": "Powerplay", "middle": "Middle", "death": "Death"}
    phase_colors = {"powerplay": STEEL_BLUE, "middle": AMBER, "death": RED}

    sr_vals, avg_vals, inns_vals, ph_names = [], [], [], []
    for ph in phase_order:
        row = phases[phases["phase"] == ph]
        if row.empty:
            continue
        r = row.iloc[0]
        if pd.isna(r.get("bat_innings", float("nan"))) or r.get("bat_innings", 0) == 0:
            continue
        sr_vals.append(float(r.get("bat_sr", 0) or 0))
        avg_vals.append(float(r.get("bat_avg", 0) or 0))
        inns_vals.append(int(r.get("bat_innings", 0) or 0))
        ph_names.append(phase_labels[ph])

    phase_chart = None
    if sr_vals:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Strike Rate",
            x=ph_names, y=sr_vals,
            marker_color=[phase_colors[p] for p in phase_order[:len(ph_names)]],
            text=[f"{v:.0f}" for v in sr_vals],
            textposition="outside",
        ))
        fig.add_trace(go.Bar(
            name="Average",
            x=ph_names, y=avg_vals,
            marker_color=[phase_colors[p] for p in phase_order[:len(ph_names)]],
            opacity=0.45,
            text=[f"{v:.1f}" for v in avg_vals],
            textposition="outside",
        ))
        fig.update_layout(
            template=PSL_TEMPLATE,
            height=220,
            barmode="group",
            margin=dict(l=10, r=10, t=10, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
            yaxis=dict(title=None),
        )
        phase_chart = dcc.Graph(figure=fig, config={"displayModeBar": False})

    # Phase SR text blocks
    phase_blocks = []
    for ph in phase_order:
        row = phases[phases["phase"] == ph]
        if row.empty:
            continue
        r = row.iloc[0]
        inns = int(r.get("bat_innings", 0) or 0)
        if inns == 0:
            continue
        sr   = float(r.get("bat_sr",  0) or 0)
        avg  = float(r.get("bat_avg", 0) or 0)
        col  = phase_colors[ph]
        sr_c = GREEN if sr >= 145 else AMBER if sr >= 115 else RED
        phase_blocks.append(
            html.Div(
                style={
                    "backgroundColor": "#1A1F26",
                    "borderTop": f"3px solid {col}",
                    "borderRadius": "5px",
                    "padding": "10px 12px",
                    "flex": "1",
                    "minWidth": "120px",
                },
                children=[
                    html.Div(phase_labels[ph], style={"color": col, "fontSize": "0.70rem", "fontWeight": "700", "marginBottom": "6px"}),
                    html.Div(f"SR {sr:.0f}", style={"color": sr_c, "fontSize": "1.10rem", "fontWeight": "900"}),
                    html.Div(f"Avg {avg:.1f}  ·  {inns} inn", style={"color": TEXT_SECONDARY, "fontSize": "0.70rem"}),
                ],
            )
        )

    return html.Div(
        style=CARD,
        children=[
            html.Div("Batting", style=SECTION_HDR),
            stats_row,
            html.Div(
                style={"display": "flex", "gap": "10px", "marginBottom": "12px"},
                children=phase_blocks,
            ),
            phase_chart or html.Div(),
        ],
    )


def _render_bowling_section(career, phases: pd.DataFrame) -> html.Div:
    def _safe(row, col, fmt=".2f", fallback="—"):
        v = row.get(col, None) if isinstance(row, dict) else getattr(row, col, None)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return fallback
        if fmt:
            return f"{v:{fmt}}"
        return str(int(v))

    eco = float(career.get("bowl_economy", float("nan")) or 0)
    eco_col = GREEN if eco <= 7.0 else AMBER if eco <= 8.5 else RED
    sr = float(career.get("bowl_sr", float("nan")) or 0)

    stats_row = html.Div(
        style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "16px"},
        children=[
            _stat_box(_safe(career, "bowl_overs", fmt=".0f"), "OVERS"),
            _stat_box(_safe(career, "bowl_wickets", fmt="", fallback="0"), "WICKETS"),
            _stat_box(_safe(career, "bowl_economy"), "ECONOMY", eco_col),
            _stat_box(_safe(career, "bowl_avg", fmt=".1f"), "AVERAGE"),
            _stat_box(_safe(career, "bowl_sr",  fmt=".1f"), "BOWL SR"),
            _stat_box(_safe(career, "bowl_dot_pct"), "DOT %"),
        ],
    )

    # Phase economy blocks
    phase_order  = ["powerplay", "middle", "death"]
    phase_labels = {"powerplay": "Powerplay", "middle": "Middle", "death": "Death"}
    phase_colors = {"powerplay": STEEL_BLUE, "middle": AMBER, "death": RED}

    eco_vals, ph_names = [], []
    phase_blocks = []
    for ph in phase_order:
        row = phases[phases["phase"] == ph]
        if row.empty:
            continue
        r = row.iloc[0]
        overs = float(r.get("bowl_overs", 0) or 0)
        if overs == 0 or pd.isna(r.get("bowl_economy", float("nan"))):
            continue
        e = float(r.get("bowl_economy", 0) or 0)
        eco_vals.append(e)
        ph_names.append(phase_labels[ph])
        col   = phase_colors[ph]
        eco_c = GREEN if e <= 7.0 else AMBER if e <= 8.5 else RED
        wkts  = int(r.get("bowl_wickets", 0) or 0)
        phase_blocks.append(
            html.Div(
                style={
                    "backgroundColor": "#1A1F26",
                    "borderTop": f"3px solid {col}",
                    "borderRadius": "5px",
                    "padding": "10px 12px",
                    "flex": "1",
                    "minWidth": "120px",
                },
                children=[
                    html.Div(phase_labels[ph], style={"color": col, "fontSize": "0.70rem", "fontWeight": "700", "marginBottom": "6px"}),
                    html.Div(f"Eco {e:.2f}", style={"color": eco_c, "fontSize": "1.10rem", "fontWeight": "900"}),
                    html.Div(f"{overs:.0f} overs  ·  {wkts} wkts", style={"color": TEXT_SECONDARY, "fontSize": "0.70rem"}),
                ],
            )
        )

    # Economy bar chart
    eco_chart = None
    if eco_vals:
        fig = go.Figure(go.Bar(
            x=ph_names, y=eco_vals,
            marker_color=[phase_colors[p] for p in phase_order[:len(ph_names)]],
            text=[f"{v:.2f}" for v in eco_vals],
            textposition="outside",
        ))
        fig.add_hline(y=8.0, line_dash="dash", line_color=AMBER, annotation_text="T20 avg 8.0", annotation_font_size=10)
        fig.update_layout(
            template=PSL_TEMPLATE,
            height=200,
            margin=dict(l=10, r=10, t=10, b=30),
            yaxis=dict(title="Economy", range=[0, max(eco_vals) * 1.25 if eco_vals else 12]),
        )
        eco_chart = dcc.Graph(figure=fig, config={"displayModeBar": False})

    return html.Div(
        style=CARD,
        children=[
            html.Div("Bowling", style=SECTION_HDR),
            stats_row,
            html.Div(
                style={"display": "flex", "gap": "10px", "marginBottom": "12px"},
                children=phase_blocks,
            ),
            eco_chart or html.Div(),
        ],
    )


def _render_recent_form(player: str, role: str, bb: pd.DataFrame) -> html.Div | None:
    """Last 8 innings (batter) or spells (bowler) as mini performance bars."""
    is_bowler = role in ("Bowler",)
    is_batter = role in ("Batsman", "Wicketkeeper", "WK-Batsman", "All-rounder")

    rows = []

    if is_batter or role == "All-rounder":
        bat_df = bb[bb["batter"] == player].groupby(["match_id", "innings", "season"]).agg(
            runs    =("runs_batter", "sum"),
            balls   =("runs_batter", "count"),
            dismissed=("is_wicket", "max"),
        ).reset_index().sort_values("season", ascending=False).head(8)

        if bat_df.empty:
            return None

        for _, r in bat_df.iterrows():
            runs  = int(r["runs"])
            balls = int(r["balls"])
            sr    = round(runs / balls * 100, 1) if balls > 0 else 0.0
            out   = bool(r["dismissed"])
            bar_w = min(100, int(runs * 2))  # rough visual scale (50 runs = full bar)
            bar_c = RED if runs < 15 else AMBER if runs < 30 else GREEN

            rows.append(html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "50px 200px 60px 60px 80px",
                    "gap": "8px",
                    "alignItems": "center",
                    "padding": "5px 0",
                    "borderBottom": f"1px solid {BORDER_COLOR}",
                },
                children=[
                    html.Div(
                        f"{'out' if out else 'not out'}",
                        style={"color": RED if out else GREEN, "fontSize": "0.68rem", "fontWeight": "700"},
                    ),
                    html.Div(
                        style={"backgroundColor": "#1E252D", "borderRadius": "3px", "height": "8px"},
                        children=[html.Div(style={"width": f"{bar_w}%", "backgroundColor": bar_c, "height": "8px", "borderRadius": "3px"})],
                    ),
                    html.Span(str(runs), style={"color": TEXT_PRIMARY, "fontWeight": "800", "fontSize": "0.88rem"}),
                    html.Span(f"{balls}b", style={"color": TEXT_SECONDARY, "fontSize": "0.75rem"}),
                    html.Span(f"SR {sr:.0f}", style={"color": TEXT_SECONDARY, "fontSize": "0.72rem"}),
                ],
            ))

    if is_bowler or role == "All-rounder":
        bowl_df = bb[bb["bowler"] == player].groupby(["match_id", "innings", "season"]).agg(
            balls  =("runs_total", "count"),
            runs   =("runs_total", "sum"),
            wickets=("is_wicket", "sum"),
        ).reset_index()
        bowl_df["economy"] = bowl_df.apply(
            lambda r: r["runs"] / (r["balls"] / 6) if r["balls"] > 0 else 0.0, axis=1
        )
        bowl_df = bowl_df.sort_values("season", ascending=False).head(8)

        if not bowl_df.empty and not rows:
            for _, r in bowl_df.iterrows():
                overs = int(r["balls"]) // 6
                balls = int(r["balls"]) % 6
                wkts  = int(r["wickets"])
                eco   = float(r["economy"])
                bar_w = min(100, max(0, int((10 - eco) * 10)))  # lower eco = wider green bar
                bar_c = GREEN if eco <= 7.0 else AMBER if eco <= 9.0 else RED
                wkt_c = GREEN if wkts >= 2 else AMBER if wkts == 1 else TEXT_SECONDARY

                rows.append(html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "60px 200px 80px 70px",
                        "gap": "8px",
                        "alignItems": "center",
                        "padding": "5px 0",
                        "borderBottom": f"1px solid {BORDER_COLOR}",
                    },
                    children=[
                        html.Div(
                            f"{overs}.{balls} ov",
                            style={"color": TEXT_SECONDARY, "fontSize": "0.72rem"},
                        ),
                        html.Div(
                            style={"backgroundColor": "#1E252D", "borderRadius": "3px", "height": "8px"},
                            children=[html.Div(style={"width": f"{bar_w}%", "backgroundColor": bar_c, "height": "8px", "borderRadius": "3px"})],
                        ),
                        html.Span(
                            f"{wkts} wkt{'s' if wkts != 1 else ''}",
                            style={"color": wkt_c, "fontWeight": "800", "fontSize": "0.85rem"},
                        ),
                        html.Span(f"Eco {eco:.2f}", style={"color": bar_c, "fontSize": "0.72rem", "fontWeight": "600"}),
                    ],
                ))

    if not rows:
        return None

    label = "Recent Batting Form" if is_batter and not is_bowler else (
            "Recent Bowling Spells" if is_bowler else "Recent Form")

    return html.Div(
        style=CARD,
        children=[html.Div(label, style=SECTION_HDR)] + rows,
    )


def _render_matchup_panel(player: str, role: str, mm: pd.DataFrame) -> html.Div:
    """Matchup panel: bowler → top victims; batter → bowlers who dismiss them."""
    sections = []

    # As a bowler — top 6 victims (most dismissals with 8+ balls)
    as_bowler = mm[(mm["bowler"] == player) & (mm["balls"] >= 8) & (mm["dismissals"] >= 1)]
    as_bowler = as_bowler.sort_values("dismissals", ascending=False).head(6)

    if not as_bowler.empty:
        victim_rows = []
        for _, r in as_bowler.iterrows():
            dis = int(r["dismissals"])
            bls = int(r["balls"])
            sr  = float(r["sr"])
            sr_c = GREEN if sr <= 100 else AMBER if sr <= 130 else RED
            victim_rows.append(
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 60px 60px 70px",
                        "gap": "8px",
                        "alignItems": "center",
                        "padding": "6px 8px",
                        "backgroundColor": "#1A1F26",
                        "borderRadius": "4px",
                        "marginBottom": "4px",
                    },
                    children=[
                        html.Span(r["batter"], style={"color": TEXT_PRIMARY, "fontSize": "0.82rem", "fontWeight": "600"}),
                        html.Span(f"{dis} dis", style={"color": GREEN, "fontWeight": "700", "fontSize": "0.78rem"}),
                        html.Span(f"{bls}b", style={"color": TEXT_SECONDARY, "fontSize": "0.75rem"}),
                        html.Span(f"SR {sr:.0f}", style={"color": sr_c, "fontSize": "0.75rem"}),
                    ],
                )
            )
        sections.append(html.Div(
            style={"flex": "1"},
            children=[
                html.Div("Top Victims", style={**SECTION_HDR, "color": GREEN}),
            ] + victim_rows,
        ))

    # As a batter — bowlers who dismiss them most (8+ balls)
    as_batter = mm[(mm["batter"] == player) & (mm["balls"] >= 8) & (mm["dismissals"] >= 1)]
    as_batter = as_batter.sort_values("dismissals", ascending=False).head(6)

    if not as_batter.empty:
        dismisser_rows = []
        for _, r in as_batter.iterrows():
            dis = int(r["dismissals"])
            bls = int(r["balls"])
            sr  = float(r["sr"])
            sr_c = GREEN if sr >= 130 else AMBER if sr >= 100 else RED
            dismisser_rows.append(
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 60px 60px 70px",
                        "gap": "8px",
                        "alignItems": "center",
                        "padding": "6px 8px",
                        "backgroundColor": "#1A1F26",
                        "borderRadius": "4px",
                        "marginBottom": "4px",
                    },
                    children=[
                        html.Span(r["bowler"], style={"color": TEXT_PRIMARY, "fontSize": "0.82rem", "fontWeight": "600"}),
                        html.Span(f"dis {dis}x", style={"color": RED, "fontWeight": "700", "fontSize": "0.78rem"}),
                        html.Span(f"{bls}b", style={"color": TEXT_SECONDARY, "fontSize": "0.75rem"}),
                        html.Span(f"SR {sr:.0f}", style={"color": sr_c, "fontSize": "0.75rem"}),
                    ],
                )
            )
        sections.append(html.Div(
            style={"flex": "1"},
            children=[
                html.Div("Dismissed By", style={**SECTION_HDR, "color": RED}),
            ] + dismisser_rows,
        ))

    if not sections:
        return html.Div(
            style=CARD,
            children=[
                html.Div("Matchups", style=SECTION_HDR),
                html.Div("Insufficient H2H data (need 8+ balls per matchup).",
                         style={"color": TEXT_SECONDARY, "fontSize": "0.80rem", "fontStyle": "italic"}),
            ],
        )

    return html.Div(
        style=CARD,
        children=[
            html.Div("Matchup History", style=SECTION_HDR),
            html.Div(style={"display": "flex", "gap": "24px"}, children=sections),
        ],
    )
