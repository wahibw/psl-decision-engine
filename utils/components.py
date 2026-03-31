# utils/components.py
# Premium reusable Dash component builders
# PSL Decision Intelligence — UI Component Library
#
# Usage:
#   from utils.components import glass_card, player_xi_card, ring_gauge, apply_theme

from __future__ import annotations

import math
from dash import html, dcc
import plotly.graph_objects as go

from utils.theme import (
    BRAND_ORANGE, DARK_BG, TEXT_PRIMARY, TEXT_SECONDARY,
    GREEN, AMBER, RED, STEEL_BLUE,
)

# ---------------------------------------------------------------------------
# PLOTLY THEME — transparent background, premium fonts, PSL palette
# ---------------------------------------------------------------------------

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'Exo 2', Inter, Arial, sans-serif", color=TEXT_SECONDARY, size=11),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.03)",
        zerolinecolor="rgba(255,255,255,0.05)",
        tickfont=dict(family="'Rajdhani', sans-serif", size=10, color="#6B7280"),
        linecolor="rgba(255,255,255,0.04)",
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.03)",
        zerolinecolor="rgba(255,255,255,0.05)",
        tickfont=dict(family="'Rajdhani', sans-serif", size=10, color="#6B7280"),
        linecolor="rgba(255,255,255,0.04)",
    ),
    legend=dict(
        font=dict(family="'Rajdhani', sans-serif", size=10, color=TEXT_SECONDARY),
        bgcolor="rgba(20,5,54,0.6)",
        bordercolor="rgba(255,140,0,0.18)",
        borderwidth=1,
    ),
    hoverlabel=dict(
        bgcolor="#23074F",
        bordercolor="rgba(255,140,0,0.4)",
        font=dict(family="'Exo 2', sans-serif", size=12, color=TEXT_PRIMARY),
    ),
    colorway=[BRAND_ORANGE, STEEL_BLUE, GREEN, AMBER, RED, "#00E5FF", "#FF6A00"],
)


def apply_theme(fig: go.Figure) -> go.Figure:
    """Apply the PSL premium theme to any Plotly figure."""
    fig.update_layout(**PLOTLY_THEME)
    return fig


# ---------------------------------------------------------------------------
# GLASS CARD
# ---------------------------------------------------------------------------

def glass_card(
    children,
    title: str | None = None,
    glow: bool = False,
    style: dict | None = None,
    id: str | None = None,
) -> html.Div:
    """
    Premium glass-morphism card wrapper.

    Args:
        children:  Dash child components (list or single).
        title:     Optional section header text (rendered with .section-header class).
        glow:      If True, adds gold pulse animation.
        style:     Extra inline style overrides.
        id:        Optional Dash component id.
    """
    cls = "glass-card glass-card-glow" if glow else "glass-card"
    merged_style = {"padding": "20px", "marginBottom": "16px", **(style or {})}

    kids = []
    if title:
        kids.append(html.Div(title, className="section-header"))
    if isinstance(children, list):
        kids.extend(children)
    else:
        kids.append(children)

    kwargs = {"className": cls, "style": merged_style, "children": kids}
    if id:
        kwargs["id"] = id
    return html.Div(**kwargs)


# ---------------------------------------------------------------------------
# PLAYER XI CARD  (FUT-style)
# ---------------------------------------------------------------------------

# Map role keywords to CSS modifier classes
_ROLE_CLASS_MAP = {
    "pace":        "pace",
    "fast":        "pace",
    "medium-fast": "pace",
    "seam":        "pace",
    "spin":        "spin",
    "off-spin":    "spin",
    "leg-spin":    "spin",
    "slow":        "spin",
    "bat":         "bat",
    "batter":      "bat",
    "batting":     "bat",
    "all-rounder": "ar",
    "allrounder":  "ar",
    "all rounder": "ar",
    "wk":          "wk",
    "wicketkeeper":"wk",
    "keeper":      "wk",
}


def _role_css(role_type: str) -> str:
    key = role_type.lower().strip()
    return _ROLE_CLASS_MAP.get(key, "bat")


def player_xi_card(
    name: str,
    role: str,
    role_type: str = "bat",
    img_url: str | None = None,
    rating: int | None = None,
) -> html.Div:
    """
    FUT-style player card for the Playing XI grid.

    Args:
        name:      Player name (displayed below avatar).
        role:      Short role label (e.g. "WK-BAT", "PACE", "SPIN AR").
        role_type: CSS modifier key — one of: pace, spin, bat, ar, wk.
        img_url:   Optional player photo URL/path.
        rating:    Optional numeric power rating (0–100).
    """
    css_class = _role_css(role_type)

    if img_url:
        avatar = html.Img(src=img_url, className="xi-avatar", alt=name)
    else:
        initials = "".join(p[0].upper() for p in name.split()[:2])
        avatar = html.Div(initials, className="xi-avatar-initials")

    children = [avatar]

    if rating is not None:
        children.append(
            html.Div(
                str(rating),
                style={
                    "fontFamily": "'Orbitron', monospace",
                    "fontSize": "11px",
                    "fontWeight": "800",
                    "color": BRAND_ORANGE,
                    "marginBottom": "2px",
                },
            )
        )

    children += [
        html.Div(name, className="xi-name"),
        html.Div(role.upper(), className=f"xi-role {css_class}"),
    ]

    return html.Div(className="xi-card psl-card-hover", children=children)


# ---------------------------------------------------------------------------
# RING GAUGE  (SVG — no JS needed)
# ---------------------------------------------------------------------------

def ring_gauge(
    value: str,
    label: str,
    color_key: str = "humidity",
    pct: float = 0.0,
    size: int = 120,
) -> html.Div:
    """
    Circular SVG ring gauge for the Weather Hub.

    Args:
        value:     Display value string (e.g. "68%", "22°C").
        label:     Short label below value (e.g. "HUMIDITY").
        color_key: CSS stroke class — one of: humidity, dew, temp, pressure.
        pct:       Fill percentage (0–100).
        size:      Diameter in pixels (default 120).
    """
    radius = (size / 2) - 10
    circumference = 2 * math.pi * radius
    offset = circumference * (1 - max(0.0, min(100.0, pct)) / 100.0)
    cx = cy = size / 2

    svg_markup = (
        f'<svg viewBox="0 0 {size} {size}" style="width:{size}px;height:{size}px;'
        f'transform:rotate(-90deg)">'
        f'<circle class="track" cx="{cx}" cy="{cy}" r="{radius}"/>'
        f'<circle class="fill {color_key}" cx="{cx}" cy="{cy}" r="{radius}"'
        f' stroke-dasharray="{circumference:.2f}"'
        f' stroke-dashoffset="{offset:.2f}"/>'
        f'</svg>'
    )

    return html.Div(
        className="ring-gauge",
        style={"width": f"{size}px", "height": f"{size}px"},
        children=[
            # SVG ring
            html.Div(
                dangerously_allow_html=True,
                children=svg_markup,
            ),
            # Center overlay
            html.Div(
                className="ring-gauge-center",
                children=[
                    html.Span(
                        value,
                        style={
                            "fontFamily": "'Orbitron', monospace",
                            "fontSize": "16px",
                            "fontWeight": "800",
                            "color": TEXT_PRIMARY,
                            "lineHeight": "1",
                        },
                    ),
                    html.Span(
                        label,
                        style={
                            "fontFamily": "'Rajdhani', sans-serif",
                            "fontSize": "8px",
                            "fontWeight": "600",
                            "letterSpacing": "2px",
                            "textTransform": "uppercase",
                            "color": "#6B7280",
                            "marginTop": "3px",
                        },
                    ),
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# COMPARISON RADAR
# ---------------------------------------------------------------------------

def comparison_radar(
    players_data: dict[str, list[float]],
    categories: list[str] | None = None,
) -> go.Figure:
    """
    Scatterpolar radar chart for comparing player attributes.

    Args:
        players_data: {player_name: [val1, val2, ...]} — values 0–100.
        categories:   Axis labels. Defaults to generic attribute names.
    """
    if categories is None:
        n = max(len(v) for v in players_data.values()) if players_data else 6
        categories = [f"Attr {i+1}" for i in range(n)]

    default_colors = [BRAND_ORANGE, STEEL_BLUE, GREEN, AMBER, RED, "#00E5FF"]
    fig = go.Figure()

    for i, (name, stats) in enumerate(players_data.items()):
        color = default_colors[i % len(default_colors)]
        # Parse hex to rgb for fill
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)

        r_vals = list(stats) + [stats[0]]
        theta_vals = list(categories) + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=r_vals,
            theta=theta_vals,
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.10)",
            line=dict(color=color, width=2),
            name=name,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=False, range=[0, 100]),
            angularaxis=dict(
                tickfont=dict(family="'Rajdhani', sans-serif", size=9, color="#6B7280"),
                linecolor="rgba(255,255,255,0.06)",
            ),
            gridshape="linear",
        ),
        showlegend=True,
        **PLOTLY_THEME,
    )
    return fig


# ---------------------------------------------------------------------------
# WAGON WHEEL OVERLAY
# ---------------------------------------------------------------------------

def wagon_wheel_overlay(
    zone_runs: list[float] | None = None,
    zones: list[str] | None = None,
) -> go.Figure:
    """
    Plotly polar bar chart intended to overlay a 3D wagon-wheel PNG.
    Use inside a .wagon-wheel-container with absolute positioning.
    """
    if zones is None:
        zones = ["Fine Leg", "Sq Leg", "Mid Wkt", "Long On",
                 "Long Off", "Cover", "Point", "3rd Man"]
    if zone_runs is None:
        zone_runs = [0] * len(zones)

    colors = [
        "#F44336", "#FF8C00", "#00E5FF", "#4CAF50",
        "#FFD700", "#FF8C00", "#8B5CF6", "#00BCD4",
    ]

    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=zone_runs,
        theta=zones,
        marker_color=colors[:len(zones)],
        opacity=0.75,
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=False),
            angularaxis=dict(
                tickfont=dict(family="'Rajdhani', sans-serif", size=9, color="#9CA3AF"),
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig
