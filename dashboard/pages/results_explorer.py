"""
Pre-Computed Results Explorer
==============================
Browse the 895 pre-computed simulation scenarios with interactive filters.
"""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

from components.plots import (
    build_coverage_vs_delta,
    build_coverage_heatmap,
    build_width_bar_chart,
    build_accuracy_ranking,
)
from engine.data_loader import load_competitor_data

dash.register_page(__name__, path="/results", name="Results",
                   title="AMRI — Results Explorer")


def layout(**kwargs):
    df = load_competitor_data()
    dgps = sorted(df["dgp"].unique().tolist()) if not df.empty else []
    methods = sorted(df["method"].unique().tolist()) if not df.empty else []
    n_values = sorted(df["n"].unique().tolist()) if not df.empty else []

    return dbc.Container([
        html.H3("Pre-Computed Results Explorer", className="mb-2"),
        html.P(
            f"Browse {len(df)} pre-computed scenarios across {len(dgps)} DGPs, "
            f"{len(methods)} methods, and {len(n_values)} sample sizes.",
            className="text-muted mb-4",
        ),

        # Filter bar
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("DGPs", className="form-label"),
                        dcc.Dropdown(
                            id="res-dgp-filter",
                            options=[{"label": d, "value": d} for d in dgps],
                            value=dgps[:3] if len(dgps) >= 3 else dgps,
                            multi=True,
                            placeholder="Select DGPs...",
                        ),
                    ], md=4),
                    dbc.Col([
                        dbc.Label("Methods", className="form-label"),
                        dcc.Dropdown(
                            id="res-method-filter",
                            options=[{"label": m.replace("_", " "), "value": m} for m in methods],
                            value=["Naive_OLS", "Sandwich_HC3", "AKS_Adaptive", "AMRI_v2"],
                            multi=True,
                            placeholder="Select methods...",
                        ),
                    ], md=4),
                    dbc.Col([
                        dbc.Label("Sample Sizes", className="form-label"),
                        dcc.Dropdown(
                            id="res-n-filter",
                            options=[{"label": str(n), "value": n} for n in n_values],
                            value=n_values,
                            multi=True,
                            placeholder="Select sample sizes...",
                        ),
                    ], md=4),
                ]),
            ]),
        ], className="mb-4"),

        # Row 1: Coverage vs delta + Heatmap
        dbc.Row([
            dbc.Col(
                dbc.Card(dbc.CardBody(dcc.Graph(id="res-coverage-delta"))),
                md=7,
            ),
            dbc.Col(
                dbc.Card(dbc.CardBody(dcc.Graph(id="res-heatmap"))),
                md=5,
            ),
        ], className="mb-3 g-3"),

        # Row 2: Width + Accuracy
        dbc.Row([
            dbc.Col(
                dbc.Card(dbc.CardBody(dcc.Graph(id="res-width-chart"))),
                md=6,
            ),
            dbc.Col(
                dbc.Card(dbc.CardBody(dcc.Graph(id="res-accuracy-chart"))),
                md=6,
            ),
        ], className="mb-3 g-3"),
    ], fluid=True)


@callback(
    Output("res-coverage-delta", "figure"),
    Output("res-heatmap", "figure"),
    Output("res-width-chart", "figure"),
    Output("res-accuracy-chart", "figure"),
    Input("res-dgp-filter", "value"),
    Input("res-method-filter", "value"),
    Input("res-n-filter", "value"),
)
def update_results(dgps, methods, n_values):
    df = load_competitor_data()
    if df.empty:
        empty = {"data": [], "layout": {"title": "No data available"}}
        return empty, empty, empty, empty

    # Filter data
    filtered = df.copy()
    if dgps:
        filtered = filtered[filtered["dgp"].isin(dgps)]
    if methods:
        filtered = filtered[filtered["method"].isin(methods)]
    if n_values:
        filtered = filtered[filtered["n"].isin(n_values)]

    # Coverage vs delta
    fig_cov = build_coverage_vs_delta(filtered, methods=methods,
                                       title="Coverage vs Delta")

    # Heatmap (use full filter)
    fig_heat = build_coverage_heatmap(filtered, title="Coverage Heatmap")

    # Width — compute per-method averages for bar chart
    width_data = []
    for m in (methods or filtered["method"].unique()):
        mdf = filtered[filtered["method"] == m]
        if not mdf.empty and "avg_width" in mdf.columns:
            width_data.append({
                "method": m,
                "avg_width": mdf["avg_width"].mean(),
            })
    fig_width = build_width_bar_chart(width_data, title="Average CI Width")

    # Accuracy — compute per-method averages
    acc_data = []
    for m in (methods or filtered["method"].unique()):
        mdf = filtered[filtered["method"] == m]
        if not mdf.empty:
            cov_acc = abs(mdf["coverage"].mean() - 0.95) if "coverage" in mdf.columns else None
            acc_data.append({
                "method": m,
                "coverage_accuracy": cov_acc,
            })
    fig_acc = build_accuracy_ranking(acc_data, title="Coverage Accuracy Ranking")

    return fig_cov, fig_heat, fig_width, fig_acc
