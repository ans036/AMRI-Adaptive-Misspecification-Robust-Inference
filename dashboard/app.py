"""
AMRI Interactive Dashboard
===========================
Main entry point for the Plotly Dash application.
Run: python dashboard/app.py
Open: http://localhost:8050
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so we can import src/ and amri/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import diskcache
from dash import Dash, html, dcc, page_container, DiskcacheManager
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeSwitchAIO

from components.navbar import create_navbar

# ---------------------------------------------------------------------------
# Background callback manager (DiskCache-based, zero infrastructure)
# Use /tmp for cloud deployments (ephemeral but writable)
# ---------------------------------------------------------------------------
_cache_dir = os.environ.get(
    "DASH_CACHE_DIR",
    str(PROJECT_ROOT / "dashboard" / ".cache"),
)
cache = diskcache.Cache(_cache_dir)
background_callback_manager = DiskcacheManager(cache)

# ---------------------------------------------------------------------------
# Theme configuration
# ---------------------------------------------------------------------------
THEME_LIGHT = dbc.themes.FLATLY
THEME_DARK = dbc.themes.DARKLY
THEME_TOGGLE_ID = "theme-switch"

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------
app = Dash(
    __name__,
    use_pages=True,
    pages_folder="pages",
    external_stylesheets=[THEME_LIGHT],
    suppress_callback_exceptions=True,
    background_callback_manager=background_callback_manager,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
    title="AMRI Dashboard",
)

server = app.server  # For deployment (gunicorn dashboard.app:server)

# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
app.layout = dbc.Container(
    [
        # Theme switch (hidden, controls stylesheet)
        ThemeSwitchAIO(
            aio_id=THEME_TOGGLE_ID,
            themes=[THEME_LIGHT, THEME_DARK],
            icons={"left": "fa fa-sun", "right": "fa fa-moon"},
        ),
        # Navbar
        create_navbar(THEME_TOGGLE_ID),
        # Page content
        html.Div(page_container, className="mt-3"),
        # Footer
        html.Footer(
            dbc.Container(
                html.P(
                    [
                        "AMRI Dashboard ",
                        html.Span("| ", className="text-muted"),
                        html.Span(
                            "Adaptive Misspecification-Robust Inference",
                            className="text-muted",
                        ),
                    ],
                    className="text-center text-muted py-3 mb-0",
                ),
                fluid=True,
            ),
            className="mt-5",
        ),
    ],
    fluid=True,
    className="px-2 px-md-4",
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DASH_DEBUG", "1") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
