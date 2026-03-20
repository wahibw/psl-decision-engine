# app.py
# Dash entry point — two-mode navigation shell
# MODE 1: /prep  — Match Prep Room (pre-match brief generator)
# MODE 2: /dugout — Dugout Screen (live match intelligence)
# PAGE 3: /players — Player Profiles (accessible from both modes)

import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

from utils.theme import (
    BRAND_ORANGE, BRAND_ACCENT, DARK_BG, DARK_ALT,
    TEXT_PRIMARY, TEXT_SECONDARY, BORDER_COLOR,
)

# ---------------------------------------------------------------------------
# APP INIT
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
    title="PSL Decision Intelligence",
)

server = app.server   # for gunicorn: web: gunicorn app:server

# ---------------------------------------------------------------------------
# NAV BAR
# ---------------------------------------------------------------------------

NAV_LINK_STYLE = {
    "color": TEXT_SECONDARY,
    "textDecoration": "none",
    "fontWeight": "600",
    "fontSize": "0.85rem",
    "letterSpacing": "0.06em",
    "padding": "6px 14px",
    "borderRadius": "4px",
    "transition": "all 0.15s",
}

NAV_ACTIVE_STYLE = {
    **NAV_LINK_STYLE,
    "color": BRAND_ORANGE,
    "backgroundColor": "rgba(255,140,0,0.10)",
    "borderBottom": f"2px solid {BRAND_ORANGE}",
}


def _nav_link(label: str, href: str, emoji: str = "") -> html.A:
    return html.A(
        f"{emoji}  {label}" if emoji else label,
        href=href,
        id=f"nav-{href.strip('/') or 'home'}",
        style=NAV_LINK_STYLE,
    )


navbar = html.Div(
    style={
        "backgroundColor": DARK_ALT,
        "borderBottom": f"2px solid {BRAND_ORANGE}",
        "padding": "0 24px",
        "display": "flex",
        "alignItems": "center",
        "height": "54px",
        "position": "sticky",
        "top": "0",
        "zIndex": "1000",
        "gap": "0",
    },
    children=[
        # Brand
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "10px", "marginRight": "32px"},
            children=[
                html.Div(
                    "PSL",
                    style={
                        "backgroundColor": BRAND_ORANGE,
                        "color": "#000",
                        "fontWeight": "900",
                        "fontSize": "0.78rem",
                        "padding": "3px 7px",
                        "borderRadius": "3px",
                        "letterSpacing": "0.1em",
                    },
                ),
                html.Span(
                    "Decision Intelligence",
                    style={
                        "color": TEXT_PRIMARY,
                        "fontWeight": "700",
                        "fontSize": "0.90rem",
                        "letterSpacing": "0.04em",
                    },
                ),
            ],
        ),
        # Mode separator label
        html.Span(
            "MODE",
            style={
                "color": BORDER_COLOR,
                "fontSize": "0.70rem",
                "letterSpacing": "0.12em",
                "marginRight": "8px",
                "alignSelf": "center",
            },
        ),
        # Nav links
        dcc.Link(
            "PREP ROOM",
            href="/prep",
            id="nav-prep",
            style=NAV_LINK_STYLE,
            refresh=False,
        ),
        html.Span(
            "|",
            style={"color": BORDER_COLOR, "padding": "0 6px", "fontSize": "1rem", "alignSelf": "center"},
        ),
        dcc.Link(
            "DUGOUT",
            href="/dugout",
            id="nav-dugout",
            style=NAV_LINK_STYLE,
            refresh=False,
        ),
        html.Span(
            "|",
            style={"color": BORDER_COLOR, "padding": "0 6px", "fontSize": "1rem", "alignSelf": "center"},
        ),
        dcc.Link(
            "PLAYERS",
            href="/players",
            id="nav-players",
            style=NAV_LINK_STYLE,
            refresh=False,
        ),
        # Spacer
        html.Div(style={"flex": "1"}),
        # Live indicator (will be updated by dugout page when match running)
        html.Div(
            id="live-indicator",
            style={"display": "flex", "alignItems": "center", "gap": "6px"},
            children=[
                html.Div(
                    style={
                        "width": "8px",
                        "height": "8px",
                        "borderRadius": "50%",
                        "backgroundColor": "#555",
                    },
                    id="live-dot",
                ),
                html.Span(
                    "STANDBY",
                    id="live-status-text",
                    style={"color": TEXT_SECONDARY, "fontSize": "0.72rem", "letterSpacing": "0.10em"},
                ),
            ],
        ),
    ],
)

# ---------------------------------------------------------------------------
# ROOT LAYOUT
# ---------------------------------------------------------------------------

app.layout = html.Div(
    style={
        "backgroundColor": DARK_BG,
        "minHeight": "100vh",
        "fontFamily": "Inter, -apple-system, Arial, sans-serif",
        "color": TEXT_PRIMARY,
    },
    children=[
        dcc.Location(id="url", refresh=False),
        # Shared cross-page store: prep room writes here, dugout reads it
        dcc.Store(id="match-brief-store", storage_type="session"),
        navbar,
        # Page content rendered by dash.page_registry
        dash.page_container,
    ],
)

# ---------------------------------------------------------------------------
# NAVBAR ACTIVE STATE CALLBACK
# ---------------------------------------------------------------------------

@app.callback(
    Output("nav-prep",    "style"),
    Output("nav-dugout",  "style"),
    Output("nav-players", "style"),
    Input("url", "pathname"),
)
def _update_nav_active(pathname: str):
    prep    = NAV_ACTIVE_STYLE if pathname and pathname.startswith("/prep")    else NAV_LINK_STYLE
    dugout  = NAV_ACTIVE_STYLE if pathname and pathname.startswith("/dugout")  else NAV_LINK_STYLE
    players = NAV_ACTIVE_STYLE if pathname and pathname.startswith("/players") else NAV_LINK_STYLE
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
    app.run(debug=True, port=8050, use_reloader=False)
