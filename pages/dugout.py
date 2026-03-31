# pages/dugout.py
# MODE 2 — Dugout Screen
# Full-screen iframe embedding the Three.js 3D dugout UI
# Route: /dugout

import dash
from dash import html

dash.register_page(__name__, path="/dugout", name="Dugout")

# Iframe is now a persistent element in app.layout (never reloaded on navigation).
# This stub keeps the Dash page registry entry so the URL route /dugout is recognised.
layout = html.Div(id="dugout-page-stub")
