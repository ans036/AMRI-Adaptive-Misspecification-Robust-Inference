"""
Method Comparison & Theory Verification
=========================================
AMRI vs AKS head-to-head, theorem verification, minimax results.
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from components.plots import COLORS, TEMPLATE_LIGHT, _nice_name, _method_color
from engine.data_loader import (
    load_competitor_data,
    load_theorem_data,
    load_minimax_data,
    load_amri_v2_data,
)

dash.register_page(__name__, path="/comparison", name="Comparison",
                   title="AMRI — Method Comparison")


def _build_amri_vs_aks(df):
    """Build AMRI v2 vs AKS head-to-head comparison figure."""
    if df.empty:
        return go.Figure().update_layout(title="No data")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Coverage Comparison", "Width Comparison"],
    )

    for method, color in [("AMRI_v2", COLORS["AMRI_v2"]), ("AKS_Adaptive", COLORS["AKS_Adaptive"])]:
        mdata = df[df["method"] == method]
        if mdata.empty:
            continue
        avg = mdata.groupby("delta").agg(
            coverage=("coverage", "mean"),
            avg_width=("avg_width", "mean") if "avg_width" in mdata.columns else ("coverage", "count"),
        ).reset_index()

        fig.add_trace(go.Scatter(
            x=avg["delta"], y=avg["coverage"],
            mode="lines+markers", name=_nice_name(method),
            line=dict(color=color, width=2.5),
            marker=dict(size=8),
            legendgroup=method,
        ), row=1, col=1)

        if "avg_width" in avg.columns:
            fig.add_trace(go.Scatter(
                x=avg["delta"], y=avg["avg_width"],
                mode="lines+markers", name=_nice_name(method),
                line=dict(color=color, width=2.5),
                marker=dict(size=8),
                legendgroup=method, showlegend=False,
            ), row=1, col=2)

    fig.add_hline(y=0.95, line=dict(color="#333", dash="dash"), row=1, col=1)
    fig.update_yaxes(title_text="Coverage", row=1, col=1)
    fig.update_yaxes(title_text="Avg Width", row=1, col=2)
    fig.update_xaxes(title_text="\u03b4 (misspecification)", row=1, col=1)
    fig.update_xaxes(title_text="\u03b4 (misspecification)", row=1, col=2)

    fig.update_layout(
        template=TEMPLATE_LIGHT, height=420,
        title="AMRI v2 vs AKS Adaptive (Armstrong, Kline & Sun 2025)",
        margin=dict(t=80),
    )
    return fig


def _build_theorem_plots(theorem_data):
    """Build theorem verification visualizations."""
    figs = []

    # Theorem 1: Continuity
    if "theorem1" in theorem_data:
        df = theorem_data["theorem1"]
        fig = go.Figure()
        for col in df.columns:
            if col == "delta":
                continue
            fig.add_trace(go.Scatter(
                x=df["delta"] if "delta" in df.columns else df.index,
                y=df[col],
                mode="lines+markers",
                name=col.replace("_", " ").title(),
            ))
        fig.update_layout(
            title="Theorem 1: Coverage Continuity",
            xaxis_title="\u03b4", yaxis_title="Value",
            template=TEMPLATE_LIGHT, height=350,
        )
        figs.append(("Coverage Continuity", fig))

    # Theorem 2: Asymptotic Coverage
    if "theorem2" in theorem_data:
        df = theorem_data["theorem2"]
        fig = go.Figure()
        if "cov_v2" in df.columns:
            for col, name, color in [
                ("cov_v2", "AMRI v2", COLORS["AMRI_v2"]),
                ("cov_naive", "Naive OLS", COLORS["Naive_OLS"]),
                ("cov_hc3", "HC3", COLORS["Sandwich_HC3"]),
            ]:
                if col in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df["n"] if "n" in df.columns else df.index,
                        y=df[col],
                        mode="lines+markers", name=name,
                        line=dict(color=color, width=2),
                    ))
        fig.add_hline(y=0.95, line=dict(color="#333", dash="dash"))
        fig.update_layout(
            title="Theorem 2: Asymptotic Coverage Validity",
            xaxis_title="Sample Size (n)", yaxis_title="Coverage",
            template=TEMPLATE_LIGHT, height=350,
        )
        figs.append(("Asymptotic Coverage", fig))

    # Theorem 3: Efficiency
    if "theorem3" in theorem_data:
        df = theorem_data["theorem3"]
        fig = go.Figure()
        for col in df.columns:
            if col in ("n", "delta"):
                continue
            fig.add_trace(go.Bar(
                x=df["n"].astype(str) if "n" in df.columns else df.index.astype(str),
                y=df[col], name=col.replace("_", " ").title(),
            ))
        fig.update_layout(
            title="Theorem 3: Efficiency Under Correct Specification",
            xaxis_title="Sample Size", yaxis_title="Width Ratio",
            template=TEMPLATE_LIGHT, height=350,
            barmode="group",
        )
        figs.append(("Efficiency Bound", fig))

    return figs


def _build_minimax_plots(minimax_data):
    """Build minimax theory visualizations."""
    figs = []

    if "minimax_lower_bound" in minimax_data:
        df = minimax_data["minimax_lower_bound"]
        fig = go.Figure()
        for col in df.columns:
            if col == "n":
                continue
            fig.add_trace(go.Bar(
                x=df["n"].astype(str) if "n" in df.columns else df.index.astype(str),
                y=df[col], name=col.replace("_", " ").title(),
            ))
        fig.update_layout(
            title="Minimax Lower Bound (Le Cam Two-Point)",
            xaxis_title="n", template=TEMPLATE_LIGHT, height=350,
            barmode="group",
        )
        figs.append(("Minimax Lower Bound", fig))

    if "near_uniform_coverage" in minimax_data:
        df = minimax_data["near_uniform_coverage"]
        fig = go.Figure()
        for col in df.columns:
            if col in ("n", "delta"):
                continue
            fig.add_trace(go.Scatter(
                x=df["n"].astype(str) if "n" in df.columns else df.index.astype(str),
                y=df[col], mode="lines+markers",
                name=col.replace("_", " ").title(),
            ))
        fig.add_hline(y=0.95, line=dict(color="#333", dash="dash"))
        fig.update_layout(
            title="Near-Uniform Coverage Across All Scenarios",
            template=TEMPLATE_LIGHT, height=350,
        )
        figs.append(("Near-Uniform Coverage", fig))

    return figs


def layout(**kwargs):
    df = load_competitor_data()
    theorem_data = load_theorem_data()
    minimax_data = load_minimax_data()

    # Build theorem and minimax plot tabs
    theorem_plots = _build_theorem_plots(theorem_data)
    minimax_plots = _build_minimax_plots(minimax_data)

    theorem_tab_content = []
    for title, fig in theorem_plots:
        theorem_tab_content.append(
            dbc.Card(dbc.CardBody(dcc.Graph(figure=fig)), className="mb-3")
        )
    if not theorem_tab_content:
        theorem_tab_content = [dbc.Alert("No theorem data available.", color="info")]

    minimax_tab_content = []
    for title, fig in minimax_plots:
        minimax_tab_content.append(
            dbc.Card(dbc.CardBody(dcc.Graph(figure=fig)), className="mb-3")
        )
    if not minimax_tab_content:
        minimax_tab_content = [dbc.Alert("No minimax data available.", color="info")]

    return dbc.Container([
        html.H3("Method Comparison & Theory", className="mb-2"),
        html.P(
            "Head-to-head comparisons, theorem verification, and minimax optimality results.",
            className="text-muted mb-4",
        ),

        dbc.Tabs([
            dbc.Tab([
                dbc.Card(dbc.CardBody(
                    dcc.Graph(figure=_build_amri_vs_aks(df))
                ), className="mt-3 mb-3"),
                dbc.Card([
                    dbc.CardHeader(html.H6("Key Finding", className="mb-0")),
                    dbc.CardBody(dcc.Markdown("""
**AMRI v2 maintains near-nominal 95% coverage** across all misspecification
levels, while AKS adaptive CIs range from 87-98%. AMRI achieves this with
only marginally wider intervals — a favorable coverage-width tradeoff.

The fundamental difference: AKS optimizes *point estimation* regret (MSE),
while AMRI optimizes *interval* performance (coverage + width). AKS's own
paper (Section 4.2.1) acknowledges that "constructing valid frequentist
confidence intervals for shrinkage estimators is an open area of research."
                    """)),
                ], className="mb-3"),
            ], label="AMRI vs AKS"),

            dbc.Tab(theorem_tab_content, label="Theorem Verification"),

            dbc.Tab(minimax_tab_content, label="Minimax Results"),
        ]),
    ], fluid=True)
