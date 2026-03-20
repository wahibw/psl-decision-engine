# utils/theme.py
# Colours, Plotly template, and shared Dash UI components
# Dark orange theme — PSL Decision Intelligence

from dash import html
import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# COLOUR PALETTE
# ---------------------------------------------------------------------------

BRAND_ORANGE   = "#FF8C00"
BRAND_ACCENT   = "#FFA726"
STEEL_BLUE     = "#4A90D9"
DARK_BG        = "#0E1117"
DARK_ALT       = "#1A1F26"
TEXT_PRIMARY   = "#E8F5E9"
TEXT_SECONDARY = "#A0AEC0"
BORDER_COLOR   = "#2D3A42"
GREEN          = "#4CAF50"
AMBER          = "#FFC107"
RED            = "#F44336"

PSL_TEAM_COLORS = {
    "Karachi Kings":           "#00A651",
    "Lahore Qalandars":        "#00AEEF",
    "Peshawar Zalmi":          "#F7941D",
    "Quetta Gladiators":       "#8B0000",
    "Islamabad United":        "#EE1C25",
    "Multan Sultans":          "#6A0DAD",
    "Pindiz":                  "#1565C0",
    "Hyderabad Houston Kings": "#F57F17",
}

PSL_2026_TEAMS = [
    "Karachi Kings",
    "Lahore Qalandars",
    "Peshawar Zalmi",
    "Quetta Gladiators",
    "Islamabad United",
    "Multan Sultans",
    "Pindiz",
    "Hyderabad Houston Kings",
]

# Priority colours used across engines and UI
PRIORITY_COLORS = {
    "CRITICAL": RED,
    "WARNING":  AMBER,
    "INFO":     GREEN,
}

# Danger level colours (partnership, batting order)
DANGER_COLORS = {
    "High":      RED,
    "Medium":    AMBER,
    "Low":       GREEN,
    "Growing":   AMBER,
    "Dangerous": RED,
    "Critical":  RED,
}

# Confidence colours
CONFIDENCE_COLORS = {
    "High":   GREEN,
    "Medium": AMBER,
    "Low":    TEXT_SECONDARY,
}

# ---------------------------------------------------------------------------
# PLOTLY TEMPLATE
# ---------------------------------------------------------------------------

_PSL_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_ALT,
        font=dict(color=TEXT_PRIMARY, family="Inter, Arial, sans-serif", size=12),
        title=dict(font=dict(color=TEXT_PRIMARY, size=15), x=0.01),
        xaxis=dict(
            gridcolor=BORDER_COLOR,
            linecolor=BORDER_COLOR,
            tickcolor=TEXT_SECONDARY,
            tickfont=dict(color=TEXT_SECONDARY, size=11),
            zerolinecolor=BORDER_COLOR,
        ),
        yaxis=dict(
            gridcolor=BORDER_COLOR,
            linecolor=BORDER_COLOR,
            tickcolor=TEXT_SECONDARY,
            tickfont=dict(color=TEXT_SECONDARY, size=11),
            zerolinecolor=BORDER_COLOR,
        ),
        legend=dict(
            bgcolor=DARK_ALT,
            bordercolor=BORDER_COLOR,
            borderwidth=1,
            font=dict(color=TEXT_PRIMARY, size=11),
        ),
        colorway=[BRAND_ORANGE, STEEL_BLUE, GREEN, AMBER, RED, BRAND_ACCENT],
        hoverlabel=dict(
            bgcolor=DARK_ALT,
            bordercolor=BORDER_COLOR,
            font=dict(color=TEXT_PRIMARY, size=12),
        ),
        margin=dict(l=40, r=20, t=40, b=40),
    )
)

pio.templates["psl_dark"] = _PSL_TEMPLATE
pio.templates.default = "psl_dark"


# ---------------------------------------------------------------------------
# SHARED STYLES (inline dicts — consistent across all pages)
# ---------------------------------------------------------------------------

CARD_STYLE = {
    "backgroundColor": DARK_ALT,
    "border":          f"1px solid {BORDER_COLOR}",
    "borderRadius":    "8px",
    "padding":         "16px",
    "marginBottom":    "16px",
}

CARD_HEADER_STYLE = {
    "color":          BRAND_ORANGE,
    "fontSize":       "13px",
    "fontWeight":     "600",
    "letterSpacing":  "0.08em",
    "textTransform":  "uppercase",
    "marginBottom":   "12px",
    "borderBottom":   f"1px solid {BORDER_COLOR}",
    "paddingBottom":  "8px",
}

STAT_VALUE_STYLE = {
    "color":      TEXT_PRIMARY,
    "fontSize":   "24px",
    "fontWeight": "700",
    "lineHeight": "1.1",
}

STAT_LABEL_STYLE = {
    "color":    TEXT_SECONDARY,
    "fontSize": "11px",
    "marginTop": "2px",
}

BADGE_BASE = {
    "display":       "inline-block",
    "padding":       "2px 8px",
    "borderRadius":  "4px",
    "fontSize":      "11px",
    "fontWeight":    "600",
    "letterSpacing": "0.04em",
}

SITUATION_READ_STYLE = {
    "fontSize":   "20px",
    "fontWeight": "700",
    "lineHeight": "1.3",
    "padding":    "16px",
    "borderRadius": "6px",
}

TABLE_HEADER_STYLE = {
    "backgroundColor": DARK_BG,
    "color":           BRAND_ORANGE,
    "fontSize":        "11px",
    "fontWeight":      "600",
    "textTransform":   "uppercase",
    "letterSpacing":   "0.06em",
    "padding":         "8px 12px",
    "borderBottom":    f"1px solid {BORDER_COLOR}",
}

TABLE_CELL_STYLE = {
    "color":        TEXT_PRIMARY,
    "fontSize":     "13px",
    "padding":      "8px 12px",
    "borderBottom": f"1px solid {BORDER_COLOR}",
}

INPUT_STYLE = {
    "backgroundColor": DARK_BG,
    "color":           TEXT_PRIMARY,
    "border":          f"1px solid {BORDER_COLOR}",
    "borderRadius":    "4px",
    "padding":         "8px 12px",
    "fontSize":        "13px",
    "width":           "100%",
}

BUTTON_PRIMARY = {
    "backgroundColor": BRAND_ORANGE,
    "color":           DARK_BG,
    "border":          "none",
    "borderRadius":    "4px",
    "padding":         "10px 24px",
    "fontSize":        "13px",
    "fontWeight":      "700",
    "cursor":          "pointer",
    "letterSpacing":   "0.05em",
    "textTransform":   "uppercase",
}

BUTTON_SECONDARY = {
    "backgroundColor": "transparent",
    "color":           BRAND_ORANGE,
    "border":          f"1px solid {BRAND_ORANGE}",
    "borderRadius":    "4px",
    "padding":         "8px 20px",
    "fontSize":        "13px",
    "fontWeight":      "600",
    "cursor":          "pointer",
}

BUTTON_DANGER = {
    **BUTTON_PRIMARY,
    "backgroundColor": RED,
    "color":           TEXT_PRIMARY,
}


# ---------------------------------------------------------------------------
# SHARED UI COMPONENT BUILDERS
# ---------------------------------------------------------------------------

def card(children, style_override=None):
    """Dark card container used throughout both modes."""
    style = {**CARD_STYLE, **(style_override or {})}
    return html.Div(children, style=style)


def card_header(title):
    """Orange uppercase section header inside a card."""
    return html.Div(title, style=CARD_HEADER_STYLE)


def stat_box(value, label, color=None):
    """Single stat: large value + small label."""
    return html.Div([
        html.Div(str(value), style={**STAT_VALUE_STYLE, "color": color or TEXT_PRIMARY}),
        html.Div(label,      style=STAT_LABEL_STYLE),
    ], style={"textAlign": "center"})


def badge(text, color):
    """Coloured pill badge — used for danger ratings, roles, confidence."""
    return html.Span(text, style={
        **BADGE_BASE,
        "backgroundColor": color + "22",   # 13% opacity fill
        "color":           color,
        "border":          f"1px solid {color}44",
    })


def danger_badge(level):
    """Badge pre-wired to DANGER_COLORS."""
    color = DANGER_COLORS.get(level, TEXT_SECONDARY)
    return badge(level, color)


def confidence_badge(level):
    """Badge pre-wired to CONFIDENCE_COLORS."""
    color = CONFIDENCE_COLORS.get(level, TEXT_SECONDARY)
    return badge(level, color)


def priority_badge(level):
    """Badge pre-wired to PRIORITY_COLORS (CRITICAL / WARNING / INFO)."""
    color = PRIORITY_COLORS.get(level, TEXT_SECONDARY)
    return badge(level, color)


def team_badge(team_name):
    """Badge coloured with the team's official PSL colour."""
    color = PSL_TEAM_COLORS.get(team_name, BRAND_ORANGE)
    return badge(team_name, color)


def situation_read_box(message, priority):
    """
    The single-sentence situation read displayed prominently in the dugout screen.
    Priority: 'CRITICAL' (red) | 'WARNING' (amber) | 'INFO' (green)
    """
    color = PRIORITY_COLORS.get(priority, GREEN)
    return html.Div(
        message,
        style={
            **SITUATION_READ_STYLE,
            "backgroundColor": color + "18",
            "color":           color,
            "border":          f"1px solid {color}44",
        }
    )


def divider():
    """Horizontal rule in border colour."""
    return html.Hr(style={"borderColor": BORDER_COLOR, "margin": "16px 0"})


def section_title(text):
    """Page-level section heading."""
    return html.H3(text, style={
        "color":         TEXT_PRIMARY,
        "fontSize":      "16px",
        "fontWeight":    "600",
        "marginBottom":  "12px",
        "marginTop":     "4px",
    })


def info_row(label, value, value_color=None):
    """Horizontal key/value row inside a card."""
    return html.Div([
        html.Span(label, style={"color": TEXT_SECONDARY, "fontSize": "12px", "minWidth": "140px", "display": "inline-block"}),
        html.Span(str(value), style={"color": value_color or TEXT_PRIMARY, "fontSize": "13px", "fontWeight": "500"}),
    ], style={"display": "flex", "alignItems": "center", "padding": "4px 0"})


def weather_icon(condition):
    """Plain-text weather icons — no external icon dependency."""
    icons = {
        "sunny":     "☀",
        "cloudy":    "☁",
        "rain":      "🌧",
        "dew":       "💧",
        "wind":      "💨",
        "temp":      "🌡",
        "humidity":  "💧",
        "warning":   "⚠",
        "critical":  "🔴",
        "ok":        "🟢",
    }
    return icons.get(condition, "")


def empty_state(message="No data available"):
    """Placeholder shown when a section has no data yet."""
    return html.Div(message, style={
        "color":      TEXT_SECONDARY,
        "fontSize":   "13px",
        "fontStyle":  "italic",
        "padding":    "24px",
        "textAlign":  "center",
    })
