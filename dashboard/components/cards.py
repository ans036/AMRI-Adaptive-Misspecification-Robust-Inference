"""Reusable card components for the dashboard."""

import dash_bootstrap_components as dbc
from dash import html


def stat_card(title: str, value: str, subtitle: str = "",
              color: str = "primary") -> dbc.Card:
    """Summary statistic card for the overview page."""
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="stat-label text-muted mb-1"),
            html.H2(value, className="stat-value", style={"color": f"var(--bs-{color})"}),
            html.P(subtitle, className="text-muted mb-0 small") if subtitle else None,
        ]),
        className="stat-card h-100",
    )


def info_card(title: str, children, icon: str = None) -> dbc.Card:
    """Generic info card with optional icon header."""
    header_items = []
    if icon:
        header_items.append(html.I(className=f"fas {icon} me-2"))
    header_items.append(title)

    return dbc.Card([
        dbc.CardHeader(html.H5(header_items, className="mb-0")),
        dbc.CardBody(children),
    ], className="h-100")
