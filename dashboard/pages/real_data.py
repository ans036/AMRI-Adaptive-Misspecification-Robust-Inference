"""
Real Data Explorer
===================
Explore AMRI performance on 61 real-world datasets.
"""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np

from components.plots import build_real_data_scatter, TEMPLATE_LIGHT, COLORS, _nice_name
from engine.data_loader import load_real_data_results

dash.register_page(__name__, path="/real-data", name="Real Data",
                   title="AMRI — Real Data Validation")


def layout(**kwargs):
    df = load_real_data_results()
    datasets = sorted(df["dataset"].unique().tolist()) if not df.empty and "dataset" in df.columns else []

    return dbc.Container([
        html.H3("Real-World Data Validation", className="mb-2"),
        html.P(
            f"AMRI tested on {len(datasets)} real-world datasets with 50,000-rep "
            "bootstrap ground truth.",
            className="text-muted mb-4",
        ),

        # Row 1: SE scatter + SE ratio histogram
        dbc.Row([
            dbc.Col(
                dbc.Card(dbc.CardBody(dcc.Graph(id="rd-se-scatter"))),
                md=6,
            ),
            dbc.Col(
                dbc.Card(dbc.CardBody(dcc.Graph(id="rd-ratio-hist"))),
                md=6,
            ),
        ], className="mb-3 g-3"),

        # Dataset selector
        dbc.Card([
            dbc.CardBody([
                dbc.Label("Select a Dataset", className="form-label"),
                dbc.Select(
                    id="rd-dataset-select",
                    options=[{"label": d, "value": d} for d in datasets],
                    value=datasets[0] if datasets else None,
                ),
            ]),
        ], className="mb-3"),

        # Row 2: Dataset detail
        html.Div(id="rd-detail"),

        # Row 3: Coverage comparison across all datasets
        dbc.Card([
            dbc.CardHeader(html.H6("Coverage Comparison Across All Datasets", className="mb-0")),
            dbc.CardBody(dcc.Graph(id="rd-coverage-comparison")),
        ], className="mb-3"),
    ], fluid=True)


@callback(
    Output("rd-se-scatter", "figure"),
    Output("rd-ratio-hist", "figure"),
    Output("rd-coverage-comparison", "figure"),
    Input("rd-dataset-select", "value"),  # trigger on page load
)
def update_overview_plots(_):
    df = load_real_data_results()
    if df.empty:
        empty = go.Figure().update_layout(title="No data available")
        return empty, empty, empty

    # SE scatter
    fig_scatter = build_real_data_scatter(df, title="SE Naive vs HC3 Across Datasets")

    # SE ratio histogram
    if "se_ratio" in df.columns:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df["se_ratio"], nbinsx=20,
            marker_color="#2166AC", opacity=0.8,
        ))
        fig_hist.add_vline(x=1.0, line=dict(color="#333", width=2, dash="dash"),
                           annotation_text="R=1 (no misspec)")
        fig_hist.update_layout(
            title="Distribution of SE Ratios (HC3/Naive)",
            xaxis_title="SE Ratio", yaxis_title="Count",
            template=TEMPLATE_LIGHT, height=400,
        )
    else:
        fig_hist = go.Figure().update_layout(title="SE ratio data not available")

    # Coverage comparison bar chart
    cov_cols = [c for c in df.columns if "coverage" in c.lower() or "cov" in c.lower()]
    if cov_cols:
        fig_cov = go.Figure()
        for col in cov_cols:
            nice = col.replace("_boot_coverage", "").replace("_coverage", "").replace("_", " ").title()
            fig_cov.add_trace(go.Box(
                y=df[col].dropna(), name=nice,
                boxmean=True,
            ))
        fig_cov.add_hline(y=0.95, line=dict(color="#333", dash="dash"),
                          annotation_text="Nominal 95%")
        fig_cov.update_layout(
            title="Bootstrap Coverage Distribution by Method",
            yaxis_title="Coverage",
            template=TEMPLATE_LIGHT, height=400,
        )
    else:
        fig_cov = go.Figure().update_layout(title="Coverage data not available")

    return fig_scatter, fig_hist, fig_cov


@callback(
    Output("rd-detail", "children"),
    Input("rd-dataset-select", "value"),
)
def update_detail(dataset):
    df = load_real_data_results()
    if df.empty or not dataset:
        return dbc.Alert("Select a dataset to see details.", color="info")

    row = df[df["dataset"] == dataset]
    if row.empty:
        return dbc.Alert(f"No data for {dataset}", color="warning")

    r = row.iloc[0]

    # Build detail cards
    detail_items = []
    for col in r.index:
        if col == "dataset":
            continue
        val = r[col]
        if isinstance(val, float):
            val = f"{val:.4f}" if abs(val) < 100 else f"{val:.2f}"
        detail_items.append(
            dbc.Col(
                dbc.Card(dbc.CardBody([
                    html.P(col.replace("_", " ").title(), className="text-muted small mb-1"),
                    html.H5(str(val), className="mb-0"),
                ]), className="text-center"),
                md=2, sm=4, className="mb-2",
            )
        )

    return dbc.Card([
        dbc.CardHeader(html.H6(f"Dataset: {dataset}", className="mb-0")),
        dbc.CardBody(dbc.Row(detail_items, className="g-2")),
    ], className="mb-3")
