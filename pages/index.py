# pages/index.py
# Root route "/" — redirect to /prep

import dash
from dash import dcc

dash.register_page(__name__, path="/")

layout = dcc.Location(id="index-redirect", href="/prep")
