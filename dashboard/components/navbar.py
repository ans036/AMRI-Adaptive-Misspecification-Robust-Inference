"""Top navigation bar for the AMRI dashboard."""

import dash_bootstrap_components as dbc
from dash import html


def create_navbar(theme_toggle_id: str) -> dbc.Navbar:
    """Build the top navbar with page links."""
    return dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand(
                    [
                        html.Span("AMRI", style={"fontWeight": "700"}),
                        html.Span(
                            " Dashboard",
                            className="d-none d-md-inline",
                            style={"fontWeight": "300"},
                        ),
                    ],
                    href="/",
                    className="me-auto",
                ),
                dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                dbc.Collapse(
                    dbc.Nav(
                        [
                            dbc.NavItem(dbc.NavLink("Overview", href="/")),
                            dbc.NavItem(
                                dbc.NavLink("Simulation", href="/simulation")
                            ),
                            dbc.NavItem(
                                dbc.NavLink("Results", href="/results")
                            ),
                            dbc.NavItem(
                                dbc.NavLink("Real Data", href="/real-data")
                            ),
                            dbc.NavItem(
                                dbc.NavLink("Comparison", href="/comparison")
                            ),
                            dbc.NavItem(dbc.NavLink("About", href="/about")),
                        ],
                        className="ms-auto",
                        navbar=True,
                    ),
                    id="navbar-collapse",
                    is_open=False,
                    navbar=True,
                ),
            ],
            fluid=True,
        ),
        color="primary",
        dark=True,
        className="mb-0",
        sticky="top",
    )
