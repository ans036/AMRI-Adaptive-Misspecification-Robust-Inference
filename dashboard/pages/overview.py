"""
Overview / Landing Page
========================
Key results summary, hero figure, and method explanation.
"""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

from components.cards import stat_card
from components.plots import build_coverage_vs_delta
from engine.data_loader import load_competitor_data

dash.register_page(__name__, path="/", name="Overview", title="AMRI Dashboard")


def layout(**kwargs):
    df = load_competitor_data()

    # Compute summary stats from data
    n_scenarios = len(df) if not df.empty else 0
    n_methods = df["method"].nunique() if not df.empty else 0
    amri_cov = ""
    if not df.empty and "AMRI_v2" in df["method"].values:
        amri_cov = f"{df[df['method'] == 'AMRI_v2']['coverage'].mean():.1%}"

    return dbc.Container([
        # Hero Section
        html.Div([
            html.H1("AMRI Dashboard"),
            html.P(
                "Adaptive Misspecification-Robust Inference: "
                "Interactive exploration of confidence intervals that adapt "
                "to model misspecification severity.",
                className="lead",
            ),
        ], className="hero-section"),

        # Summary Cards
        dbc.Row([
            dbc.Col(stat_card("Scenarios", str(n_scenarios), "Simulation configurations", "primary"), md=3),
            dbc.Col(stat_card("Methods", str(n_methods), "Compared head-to-head", "info"), md=3),
            dbc.Col(stat_card("Datasets", "61", "Real-world validation", "success"), md=3),
            dbc.Col(stat_card("AMRI v2 Coverage", amri_cov, "Average across all scenarios", "warning"), md=3),
        ], className="mb-4 g-3"),

        # Hero Figure
        dbc.Card([
            dbc.CardHeader(html.H5("Coverage vs Misspecification Severity", className="mb-0")),
            dbc.CardBody(
                dcc.Graph(
                    id="hero-coverage-plot",
                    figure=build_coverage_vs_delta(
                        df,
                        methods=["Naive_OLS", "Sandwich_HC3", "AKS_Adaptive", "AMRI_v2"],
                        title="How does coverage degrade as misspecification increases?",
                    ),
                    config={"displayModeBar": True, "toImageButtonOptions": {"format": "svg"}},
                )
            ),
        ], className="mb-4"),

        # Method Explanation + Quick Links
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(html.H5("The AMRI v2 Method", className="mb-0")),
                    dbc.CardBody([
                        dcc.Markdown("""
**Problem:** Standard confidence intervals either assume the model is correct
(Naive OLS — efficient but fragile) or always use robust standard errors
(HC3 — safe but wide). Neither adapts to the actual degree of misspecification.

**Solution:** AMRI v2 *blends* model-based and robust SEs using a data-driven
weight that responds to the SE ratio:

```
R  = SE_robust / SE_naive       (diagnostic ratio)
s  = |log(R)|                   (natural scale)
w  = clip((s - c1/√n) / (c2/√n - c1/√n), 0, 1)
SE = (1-w) × SE_naive + w × SE_robust
```

When the model appears correct (R ≈ 1), `w → 0` and AMRI uses the efficient
SE. When misspecification is detected (R far from 1), `w → 1` and AMRI
switches to the robust SE. The transition is **smooth** (no discontinuous jumps).
                        """, style={"fontSize": "0.95rem"}),
                    ]),
                ], className="h-100"),
                md=8,
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(html.H5("Explore", className="mb-0")),
                    dbc.CardBody([
                        html.P("Jump to a section:", className="text-muted"),
                        dbc.Nav([
                            dbc.NavItem(dbc.NavLink(
                                "Run a Simulation",
                                href="/simulation",
                                className="quick-link-btn mb-2",
                                style={"display": "block"},
                            )),
                            dbc.NavItem(dbc.NavLink(
                                "Browse Results",
                                href="/results",
                                className="quick-link-btn mb-2",
                                style={"display": "block"},
                            )),
                            dbc.NavItem(dbc.NavLink(
                                "Real Data Validation",
                                href="/real-data",
                                className="quick-link-btn mb-2",
                                style={"display": "block"},
                            )),
                            dbc.NavItem(dbc.NavLink(
                                "Method Comparison",
                                href="/comparison",
                                className="quick-link-btn mb-2",
                                style={"display": "block"},
                            )),
                            dbc.NavItem(dbc.NavLink(
                                "About AMRI",
                                href="/about",
                                className="quick-link-btn mb-2",
                                style={"display": "block"},
                            )),
                        ], vertical=True),
                    ]),
                ], className="h-100"),
                md=4,
            ),
        ], className="mb-4 g-3"),

    ], fluid=True)
