"""
Reusable Plotly figure builders for the AMRI dashboard.

All functions return a plotly.graph_objects.Figure ready for dcc.Graph.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Color scheme (matches publication figures)
# ---------------------------------------------------------------------------
COLORS = {
    "AMRI_v2": "#2166AC",
    "AMRI_v1": "#67A9CF",
    "Sandwich_HC3": "#D6604D",
    "Sandwich_HC4": "#F4A582",
    "Sandwich_HC5": "#B2182B",
    "AKS_Adaptive": "#5AB4AC",
    "Naive_OLS": "#999999",
    "Pairs_Bootstrap": "#B2ABD2",
    "Wild_Bootstrap": "#FDB863",
    "Bootstrap_t": "#E08214",
}

TEMPLATE_LIGHT = "plotly_white"
TEMPLATE_DARK = "plotly_dark"


def _method_color(method: str) -> str:
    return COLORS.get(method, "#666666")


def _nice_name(method: str) -> str:
    return method.replace("_", " ")


# ---------------------------------------------------------------------------
# 1. Forest Plot — Inference Results (THE KEY VISUALIZATION)
# ---------------------------------------------------------------------------
def build_forest_plot(results_list, theta_true, title="Inference Results"):
    """
    Horizontal CI bars for each method, centered on theta_hat.

    Parameters
    ----------
    results_list : list[dict]
        Each dict has: method, theta_hat, se, ci_low, ci_high, covers_true, width
    theta_true : float
        True parameter value (vertical reference line).
    """
    fig = go.Figure()

    methods = [r["method"] for r in results_list if r["theta_hat"] is not None]
    n_methods = len(methods)

    # Vertical reference line at theta_true
    fig.add_shape(
        type="line", x0=theta_true, x1=theta_true,
        y0=-0.5, y1=n_methods - 0.5,
        line=dict(color="#333333", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=theta_true, y=n_methods - 0.3,
        text=f"True theta = {theta_true:.3f}",
        showarrow=False, font=dict(size=10, color="#333333"),
        yshift=15,
    )

    valid_results = [r for r in results_list if r["theta_hat"] is not None]
    for i, r in enumerate(valid_results):
        covers = r["covers_true"]
        color = "#2ca02c" if covers else "#d62728"  # green if covers, red if not
        method_color = _method_color(r["method"])

        # CI bar (horizontal line)
        fig.add_trace(go.Scatter(
            x=[r["ci_low"], r["ci_high"]],
            y=[i, i],
            mode="lines",
            line=dict(color=color, width=4),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Point estimate (diamond marker)
        fig.add_trace(go.Scatter(
            x=[r["theta_hat"]],
            y=[i],
            mode="markers",
            marker=dict(
                symbol="diamond", size=12,
                color=method_color, line=dict(color="white", width=1.5),
            ),
            showlegend=False,
            hovertemplate=(
                f"<b>{_nice_name(r['method'])}</b><br>"
                f"theta_hat = {r['theta_hat']:.4f}<br>"
                f"SE = {r['se']:.4f}<br>"
                f"CI = [{r['ci_low']:.4f}, {r['ci_high']:.4f}]<br>"
                f"Width = {r['width']:.4f}<br>"
                f"Covers true: {'Yes' if covers else 'No'}"
                "<extra></extra>"
            ),
        ))

        # Annotation: SE and width on the right
        fig.add_annotation(
            x=r["ci_high"], y=i,
            text=f"  SE={r['se']:.3f}  W={r['width']:.3f}",
            showarrow=False, xanchor="left",
            font=dict(size=9, color="#666666"),
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Parameter Value",
        yaxis=dict(
            tickvals=list(range(len(valid_results))),
            ticktext=[_nice_name(r["method"]) for r in valid_results],
            autorange="reversed",
        ),
        template=TEMPLATE_LIGHT,
        height=max(350, 55 * n_methods + 100),
        margin=dict(l=140, r=160, t=60, b=50),
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# ---------------------------------------------------------------------------
# 2. Data Scatter with Fit Lines
# ---------------------------------------------------------------------------
def build_scatter_with_fit(X, Y, theta_true, intercept, title="Generated Data"):
    """Scatter plot of data with true and fitted regression lines."""
    X = np.array(X)
    Y = np.array(Y)

    fig = go.Figure()

    # Data points
    fig.add_trace(go.Scatter(
        x=X, y=Y, mode="markers",
        marker=dict(size=4, color="#4C72B0", opacity=0.4),
        name="Data",
        hovertemplate="X=%{x:.2f}<br>Y=%{y:.2f}<extra></extra>",
    ))

    # True regression line
    x_range = np.linspace(X.min(), X.max(), 100)
    y_mean = np.mean(Y)
    x_mean = np.mean(X)
    fitted_intercept = y_mean - theta_true * x_mean  # approximate

    fig.add_trace(go.Scatter(
        x=x_range, y=fitted_intercept + theta_true * x_range,
        mode="lines", line=dict(color="#d62728", width=2, dash="dash"),
        name=f"True (slope={theta_true:.3f})",
    ))

    # OLS fitted line
    fig.add_trace(go.Scatter(
        x=x_range, y=intercept + np.polyfit(X, Y, 1)[0] * x_range + (np.polyfit(X, Y, 1)[1] - np.polyfit(X, Y, 1)[0] * x_range[0] + intercept + np.polyfit(X, Y, 1)[0] * x_range[0] - np.polyfit(X, Y, 1)[1]),
        mode="lines", line=dict(color="#2166AC", width=2),
        name="OLS Fit",
        visible=False,  # Will add properly below
    ))

    # Simpler OLS fit line
    ols_slope, ols_intercept = np.polyfit(X, Y, 1)
    fig.data = fig.data[:2]  # Remove the broken trace
    fig.add_trace(go.Scatter(
        x=x_range, y=ols_intercept + ols_slope * x_range,
        mode="lines", line=dict(color="#2166AC", width=2),
        name=f"OLS Fit (slope={ols_slope:.3f})",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="X", yaxis_title="Y",
        template=TEMPLATE_LIGHT,
        height=350,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, font=dict(size=10)),
        margin=dict(l=50, r=20, t=50, b=50),
    )

    return fig


# ---------------------------------------------------------------------------
# 3. Coverage Bar Chart
# ---------------------------------------------------------------------------
def build_coverage_bar_chart(mc_results, alpha=0.05, title="Coverage by Method"):
    """Bar chart of coverage per method with nominal reference line."""
    nominal = 1 - alpha
    methods = [r["method"] for r in mc_results if r["coverage"] is not None]
    coverages = [r["coverage"] for r in mc_results if r["coverage"] is not None]
    mc_ses = [r.get("coverage_mc_se", 0) for r in mc_results if r["coverage"] is not None]

    fig = go.Figure()

    # Acceptable band
    fig.add_hrect(y0=nominal - 0.02, y1=nominal + 0.02,
                  fillcolor="green", opacity=0.08, line_width=0)

    # Nominal line
    fig.add_hline(y=nominal, line=dict(color="#333", width=1.5, dash="dash"),
                  annotation_text=f"Nominal ({nominal:.0%})",
                  annotation_position="top right")

    fig.add_trace(go.Bar(
        x=[_nice_name(m) for m in methods],
        y=coverages,
        error_y=dict(type="data", array=[1.96 * s for s in mc_ses], visible=True),
        marker_color=[_method_color(m) for m in methods],
        hovertemplate="%{x}<br>Coverage: %{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        yaxis_title="Coverage",
        yaxis=dict(range=[min(0.85, min(coverages) - 0.02) if coverages else 0.85, 1.0]),
        template=TEMPLATE_LIGHT,
        height=380,
        margin=dict(l=60, r=20, t=50, b=80),
        xaxis_tickangle=-30,
    )

    return fig


# ---------------------------------------------------------------------------
# 4. Width Comparison Bar Chart
# ---------------------------------------------------------------------------
def build_width_bar_chart(mc_results, title="Average CI Width"):
    """Bar chart of average CI width per method, sorted narrowest first."""
    valid = [r for r in mc_results if r["avg_width"] is not None]
    valid.sort(key=lambda r: r["avg_width"])

    methods = [r["method"] for r in valid]
    widths = [r["avg_width"] for r in valid]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[_nice_name(m) for m in methods],
        y=widths,
        marker_color=[_method_color(m) for m in methods],
        hovertemplate="%{x}<br>Avg Width: %{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        yaxis_title="Average CI Width",
        template=TEMPLATE_LIGHT,
        height=380,
        margin=dict(l=60, r=20, t=50, b=80),
        xaxis_tickangle=-30,
    )

    return fig


# ---------------------------------------------------------------------------
# 5. Coverage vs Delta Line Plot
# ---------------------------------------------------------------------------
def build_coverage_vs_delta(df, methods=None, dgp_filter=None,
                            n_filter=None, alpha=0.05,
                            title="Coverage vs Misspecification Severity"):
    """
    Line plot: coverage (y) vs delta (x) for each method.
    Uses pre-computed DataFrame (results_competitor_comparison.csv format).
    """
    if df.empty:
        return go.Figure().update_layout(title="No data available")

    nominal = 1 - alpha
    data = df.copy()

    if dgp_filter:
        data = data[data["dgp"].isin(dgp_filter)] if isinstance(dgp_filter, list) else data[data["dgp"] == dgp_filter]
    if n_filter:
        data = data[data["n"].isin(n_filter)] if isinstance(n_filter, list) else data[data["n"] == n_filter]
    if methods:
        data = data[data["method"].isin(methods)]

    if data.empty:
        return go.Figure().update_layout(title="No data for selected filters")

    fig = go.Figure()

    # Acceptable band
    fig.add_hrect(y0=nominal - 0.02, y1=nominal + 0.02,
                  fillcolor="green", opacity=0.06, line_width=0)
    fig.add_hline(y=nominal, line=dict(color="#333", width=1, dash="dash"))

    for method in (methods or data["method"].unique()):
        mdata = data[data["method"] == method]
        if mdata.empty:
            continue
        avg = mdata.groupby("delta")["coverage"].mean().reset_index()
        fig.add_trace(go.Scatter(
            x=avg["delta"], y=avg["coverage"],
            mode="lines+markers",
            name=_nice_name(method),
            line=dict(color=_method_color(method), width=2.5),
            marker=dict(size=7),
            hovertemplate=f"{_nice_name(method)}<br>delta=%{{x}}<br>Coverage=%{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Misspecification Severity (delta)",
        yaxis_title="Coverage",
        yaxis=dict(range=[0.85, 1.0]),
        template=TEMPLATE_LIGHT,
        height=450,
        legend=dict(font=dict(size=10)),
        margin=dict(l=60, r=20, t=60, b=50),
    )

    return fig


# ---------------------------------------------------------------------------
# 6. Coverage Heatmap
# ---------------------------------------------------------------------------
def build_coverage_heatmap(df, title="Coverage Heatmap"):
    """Heatmap: rows = methods, columns = DGPs, cells = average coverage."""
    if df.empty:
        return go.Figure().update_layout(title="No data available")

    pivot = df.pivot_table(values="coverage", index="method", columns="dgp", aggfunc="mean")

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=[_nice_name(m) for m in pivot.index],
        colorscale="RdBu",
        zmid=0.95,
        zmin=0.85,
        zmax=1.0,
        text=np.round(pivot.values, 3),
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="Method: %{y}<br>DGP: %{x}<br>Coverage: %{z:.4f}<extra></extra>",
        colorbar=dict(title="Coverage"),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        template=TEMPLATE_LIGHT,
        height=max(300, 35 * len(pivot.index) + 100),
        margin=dict(l=140, r=20, t=60, b=80),
        xaxis_tickangle=-30,
    )

    return fig


# ---------------------------------------------------------------------------
# 7. Blending Weight Histogram
# ---------------------------------------------------------------------------
def build_weight_histogram(weights, title="AMRI v2 Blending Weight Distribution"):
    """Distribution of blending weights w across B replications."""
    if not weights:
        return go.Figure().update_layout(title="No weight data")

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=weights,
        nbinsx=30,
        marker_color="#2166AC",
        opacity=0.8,
        hovertemplate="w in [%{x}]<br>Count: %{y}<extra></extra>",
    ))

    # Reference lines at w=0 and w=1
    fig.add_vline(x=0, line=dict(color="#2ca02c", width=1.5, dash="dot"),
                  annotation_text="Efficient", annotation_position="top left")
    fig.add_vline(x=1, line=dict(color="#d62728", width=1.5, dash="dot"),
                  annotation_text="Robust", annotation_position="top right")

    mean_w = np.mean(weights)
    fig.add_vline(x=mean_w, line=dict(color="#333", width=2, dash="dash"),
                  annotation_text=f"Mean={mean_w:.3f}")

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Blending Weight (w)",
        yaxis_title="Count",
        xaxis=dict(range=[-0.05, 1.05]),
        template=TEMPLATE_LIGHT,
        height=350,
        margin=dict(l=60, r=20, t=60, b=50),
    )

    return fig


# ---------------------------------------------------------------------------
# 8. Diagnostic Panel
# ---------------------------------------------------------------------------
def build_diagnostic_panel(se_ratio, weight, c1, c2, n):
    """
    Visual number line showing where |log R| falls relative to thresholds.
    """
    log_r = abs(np.log(se_ratio)) if se_ratio > 0 else 0
    lo = c1 / np.sqrt(n)
    hi = c2 / np.sqrt(n)

    fig = go.Figure()

    # Efficient zone (green)
    fig.add_shape(type="rect", x0=0, x1=lo, y0=0.3, y1=0.7,
                  fillcolor="rgba(44, 160, 44, 0.2)", line=dict(width=0))
    # Transition zone (yellow)
    fig.add_shape(type="rect", x0=lo, x1=hi, y0=0.3, y1=0.7,
                  fillcolor="rgba(255, 193, 7, 0.2)", line=dict(width=0))
    # Robust zone (red)
    x_max = max(hi * 2, log_r * 1.3, 0.5)
    fig.add_shape(type="rect", x0=hi, x1=x_max, y0=0.3, y1=0.7,
                  fillcolor="rgba(214, 39, 40, 0.15)", line=dict(width=0))

    # Threshold lines
    fig.add_shape(type="line", x0=lo, x1=lo, y0=0.2, y1=0.8,
                  line=dict(color="#2ca02c", width=2, dash="dash"))
    fig.add_shape(type="line", x0=hi, x1=hi, y0=0.2, y1=0.8,
                  line=dict(color="#d62728", width=2, dash="dash"))

    # Current |log R| marker
    fig.add_trace(go.Scatter(
        x=[log_r], y=[0.5], mode="markers+text",
        marker=dict(symbol="diamond", size=18, color="#2166AC",
                    line=dict(color="white", width=2)),
        text=[f"|log R| = {log_r:.4f}"],
        textposition="top center",
        textfont=dict(size=11, color="#2166AC"),
        showlegend=False,
        hovertemplate=f"|log R| = {log_r:.4f}<br>w = {weight:.4f}<extra></extra>",
    ))

    # Zone labels
    fig.add_annotation(x=lo / 2, y=0.15, text="Efficient", showarrow=False,
                       font=dict(size=10, color="#2ca02c"))
    fig.add_annotation(x=(lo + hi) / 2, y=0.15, text="Transition", showarrow=False,
                       font=dict(size=10, color="#FF8C00"))
    fig.add_annotation(x=hi + (x_max - hi) / 3, y=0.15, text="Robust", showarrow=False,
                       font=dict(size=10, color="#d62728"))

    # Threshold labels
    fig.add_annotation(x=lo, y=0.87, text=f"c1/sqrt(n)={lo:.4f}", showarrow=False,
                       font=dict(size=9))
    fig.add_annotation(x=hi, y=0.87, text=f"c2/sqrt(n)={hi:.4f}", showarrow=False,
                       font=dict(size=9))

    fig.update_layout(
        title=dict(text=f"AMRI Diagnostic: R={se_ratio:.3f}, w={weight:.3f}",
                   font=dict(size=13)),
        xaxis=dict(range=[0, x_max], title="|log(SE_ratio)|"),
        yaxis=dict(range=[0, 1], visible=False),
        template=TEMPLATE_LIGHT,
        height=220,
        margin=dict(l=50, r=20, t=50, b=40),
    )

    return fig


# ---------------------------------------------------------------------------
# 9. Real Data: SE_naive vs SE_HC3 scatter
# ---------------------------------------------------------------------------
def build_real_data_scatter(df, title="SE Comparison Across Datasets"):
    """Scatter: SE_naive (x) vs SE_HC3 (y) with 45-degree reference line."""
    if df.empty:
        return go.Figure().update_layout(title="No data")

    x_col = "se_naive" if "se_naive" in df.columns else df.columns[0]
    y_col = "se_hc3" if "se_hc3" in df.columns else df.columns[1]

    fig = go.Figure()

    # 45-degree line
    max_val = max(df[x_col].max(), df[y_col].max()) * 1.1
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", line=dict(color="#999", width=1, dash="dash"),
        showlegend=False, hoverinfo="skip",
    ))

    # Data points
    dataset_names = df["dataset"] if "dataset" in df.columns else df.index.astype(str)
    n_vals = df["n"] if "n" in df.columns else None

    fig.add_trace(go.Scatter(
        x=df[x_col], y=df[y_col],
        mode="markers",
        marker=dict(
            size=8 if n_vals is None else np.clip(np.log(n_vals) * 3, 5, 20),
            color=df["se_ratio"] if "se_ratio" in df.columns else "#2166AC",
            colorscale="RdBu_r",
            showscale=True if "se_ratio" in df.columns else False,
            colorbar=dict(title="SE Ratio") if "se_ratio" in df.columns else None,
            line=dict(width=0.5, color="white"),
        ),
        text=dataset_names,
        hovertemplate="%{text}<br>SE_naive=%{x:.4f}<br>SE_HC3=%{y:.4f}<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Naive SE", yaxis_title="HC3 SE",
        template=TEMPLATE_LIGHT,
        height=400,
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig


# ---------------------------------------------------------------------------
# 10. Coverage Accuracy Ranking
# ---------------------------------------------------------------------------
def build_accuracy_ranking(mc_results, alpha=0.05,
                           title="Coverage Accuracy |cov - nominal|"):
    """Horizontal bar chart ranking methods by coverage accuracy."""
    valid = [r for r in mc_results if r.get("coverage_accuracy") is not None]
    valid.sort(key=lambda r: r["coverage_accuracy"])

    methods = [r["method"] for r in valid]
    accs = [r["coverage_accuracy"] for r in valid]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[_nice_name(m) for m in methods],
        x=accs,
        orientation="h",
        marker_color=[_method_color(m) for m in methods],
        hovertemplate="%{y}<br>Accuracy: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="|Coverage - Nominal|",
        template=TEMPLATE_LIGHT,
        height=max(300, 35 * len(methods) + 80),
        margin=dict(l=140, r=20, t=50, b=50),
    )

    return fig
