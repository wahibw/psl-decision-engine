# pages/prep_room.py
# MODE 1 — Match Prep Room
# Full-screen iframe embedding the Three.js 3D prep room UI
# Route: /prep

import dash
from dash import html

dash.register_page(__name__, path="/prep", name="Prep Room")

# Iframe is now a persistent element in app.layout (never reloaded on navigation).
# This stub keeps the Dash page registry entry so the URL route /prep is recognised.
layout = html.Div(id="prep-page-stub")
