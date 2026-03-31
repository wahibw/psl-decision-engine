# pages/player_profiles.py
# PAGE 3 — Player Profiles
# Full-screen iframe embedding the Three.js 3D players stadium UI
# Route: /players

import dash
from dash import html

dash.register_page(__name__, path="/players", name="Players")

# Iframe is now a persistent element in app.layout (never reloaded on navigation).
# This stub keeps the Dash page registry entry so the URL route /players is recognised.
layout = html.Div(id="players-page-stub")
