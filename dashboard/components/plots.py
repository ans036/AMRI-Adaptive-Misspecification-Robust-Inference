"""
Reusable Plotly figure builders for the AMRI dashboard.

All functions return a plotly.graph_objects.Figure ready for dcc.Graph.
"""

import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Color scheme — high-contrast, visually distinct per method
# ---------------------------------------------------------------------------
COLORS = {
    "AMRI_v2": "#1A5276",       # deep navy (flagship)
    "AMRI_v1": "#2E86C1",       # bright blue
    "Sandwich_HC3": "#C0392B",  # strong red
    "Sandwich_HC4": "#E67E22",  # orange
    "Sandwich_HC5": "#8E44AD",  # purple
    "AKS_Adaptive": "#27AE60",  # green
    "Naive_OLS": "#7F8C8D",    # gray
    "Pairs_Bootstrap": "#D4AC0D", # gold
    "Wild_Bootstrap": "#E74C3C", # bright red-orange
    "Bootstrap_t": "#16A085",   # teal
}

TEMPLATE_LIGHT = "plotly_white"
TEMPLATE_DARK = "plotly_dark"


def _method_color(method: str) -> str:
    return COLORS.get(method, "#666666")


def _nice_name(method: str) -> str:
    return method.replace("_", " ")


def _sfmt(x, digits=4):
    """Safe number formatter for plot annotations."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "N/A"
    abs_x = abs(x)
    if abs_x == 0:
        return "0"
    if abs_x >= 1e6 or abs_x < 1e-6:
        return f"{x:.2e}"
    if abs_x >= 100:
        return f"{x:.1f}"
    return f"{x:.{digits}f}"


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

    # Filter out results with non-finite values
    valid_results = [r for r in results_list
                     if r["theta_hat"] is not None
                     and np.isfinite(r.get("ci_low", 0))
                     and np.isfinite(r.get("ci_high", 0))]
    n_methods = len(valid_results)

    # Vertical reference line at theta_true (only if known)
    if theta_true is not None:
        fig.add_shape(
            type="line", x0=theta_true, x1=theta_true,
            y0=-0.5, y1=n_methods - 0.5,
            line=dict(color="#333333", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=theta_true, y=-0.5,
            text=f"True \u03b8 = {_sfmt(theta_true)}",
            showarrow=False, font=dict(size=11, color="#333333"),
            yshift=-18,
        )

    for i, r in enumerate(valid_results):
        covers = r.get("covers_true")
        # Green if covers, red if misses, method color if unknown
        if covers is True:
            color = "#27AE60"
        elif covers is False:
            color = "#C0392B"
        else:
            color = _method_color(r["method"])
        method_color = _method_color(r["method"])

        # CI bar (horizontal line)
        fig.add_trace(go.Scatter(
            x=[r["ci_low"], r["ci_high"]],
            y=[i, i],
            mode="lines",
            line=dict(color=color, width=5),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Point estimate (diamond marker)
        covers_text = ""
        if covers is True:
            covers_text = "Covers true \u03b8: Yes"
        elif covers is False:
            covers_text = "Covers true \u03b8: No"

        fig.add_trace(go.Scatter(
            x=[r["theta_hat"]],
            y=[i],
            mode="markers",
            marker=dict(
                symbol="diamond", size=13,
                color=method_color, line=dict(color="white", width=1.5),
            ),
            showlegend=False,
            hovertemplate=(
                f"<b>{_nice_name(r['method'])}</b><br>"
                f"\u03b8\u0302 = {_sfmt(r['theta_hat'])}<br>"
                f"SE = {_sfmt(r['se'])}<br>"
                f"CI = [{_sfmt(r['ci_low'])}, {_sfmt(r['ci_high'])}]<br>"
                f"Width = {_sfmt(r['width'])}"
                + (f"<br>{covers_text}" if covers_text else "")
                + "<extra></extra>"
            ),
        ))

        # Annotation: SE and width on the right
        fig.add_annotation(
            x=r["ci_high"], y=i,
            text=f"  SE={_sfmt(r['se'])}  W={_sfmt(r['width'])}",
            showarrow=False, xanchor="left",
            font=dict(size=10, color="#555555"),
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
        margin=dict(l=140, r=180, t=60, b=50),
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
            hovertemplate=f"{_nice_name(method)}<br>\u03b4 = %{{x}}<br>Coverage = %{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Misspecification Severity (\u03b4)",
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

    # Ensure enough space for all zones
    x_max = max(hi * 2.5, log_r * 1.5, 0.5)

    # Efficient zone (green)
    fig.add_shape(type="rect", x0=0, x1=lo, y0=0.3, y1=0.7,
                  fillcolor="rgba(44, 160, 44, 0.2)", line=dict(width=0))
    # Transition zone (yellow)
    fig.add_shape(type="rect", x0=lo, x1=hi, y0=0.3, y1=0.7,
                  fillcolor="rgba(255, 193, 7, 0.2)", line=dict(width=0))
    # Robust zone (red)
    fig.add_shape(type="rect", x0=hi, x1=x_max, y0=0.3, y1=0.7,
                  fillcolor="rgba(214, 39, 40, 0.12)", line=dict(width=0))

    # Threshold lines
    fig.add_shape(type="line", x0=lo, x1=lo, y0=0.2, y1=0.8,
                  line=dict(color="#2ca02c", width=2, dash="dash"))
    fig.add_shape(type="line", x0=hi, x1=hi, y0=0.2, y1=0.8,
                  line=dict(color="#d62728", width=2, dash="dash"))

    # Current |log R| marker
    fig.add_trace(go.Scatter(
        x=[log_r], y=[0.5], mode="markers",
        marker=dict(symbol="diamond", size=18, color="#2166AC",
                    line=dict(color="white", width=2)),
        showlegend=False,
        hovertemplate=(
            f"|log R| = {log_r:.4f}<br>"
            f"w = {weight:.3f}<extra></extra>"
        ),
    ))

    # Marker label (positioned to avoid overlap)
    marker_y = 0.82 if log_r < (lo + hi) / 2 else 0.82
    fig.add_annotation(
        x=log_r, y=marker_y,
        text=f"|log R| = {log_r:.3f}",
        showarrow=True, arrowhead=2, arrowsize=0.8,
        ax=0, ay=-25,
        font=dict(size=11, color="#2166AC"),
    )

    # Zone labels (at bottom, well spaced)
    fig.add_annotation(x=lo / 2, y=0.12, text="Efficient",
                       showarrow=False, font=dict(size=11, color="#2ca02c"))
    fig.add_annotation(x=(lo + hi) / 2, y=0.12, text="Transition",
                       showarrow=False, font=dict(size=11, color="#FF8C00"))
    fig.add_annotation(x=min(hi + (x_max - hi) * 0.4, x_max * 0.8),
                       y=0.12, text="Robust",
                       showarrow=False, font=dict(size=11, color="#d62728"))

    # Threshold labels (offset vertically to avoid collision)
    fig.add_annotation(
        x=lo, y=0.95,
        text=f"c\u2081/\u221an = {lo:.3f}",
        showarrow=False, font=dict(size=10, color="#2ca02c"),
        xanchor="center",
    )
    fig.add_annotation(
        x=hi, y=0.95,
        text=f"c\u2082/\u221an = {hi:.3f}",
        showarrow=False, font=dict(size=10, color="#d62728"),
        xanchor="center",
    )

    fig.update_layout(
        title=dict(
            text=f"AMRI Diagnostic: R = {se_ratio:.3f}, w = {weight:.3f}",
            font=dict(size=13),
        ),
        xaxis=dict(range=[0, x_max], title="|log(SE ratio)|"),
        yaxis=dict(range=[0, 1.05], visible=False),
        template=TEMPLATE_LIGHT,
        height=240,
        margin=dict(l=40, r=20, t=50, b=40),
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


# ---------------------------------------------------------------------------
# 11. SE Comparison — Horizontal lollipop chart showing % deviation from Naive
# ---------------------------------------------------------------------------
def build_se_comparison(results_list, title="SE Comparison Across Methods"):
    """
    Horizontal lollipop chart: each method's SE as a dot with a line
    extending from the Naive baseline. Shows % deviation clearly.
    """
    valid = [r for r in results_list
             if r["se"] is not None and np.isfinite(r["se"])]
    if not valid:
        return go.Figure().update_layout(title="No data")

    # Sort by SE (smallest first)
    valid = sorted(valid, key=lambda r: r["se"])

    naive_se = next((r["se"] for r in valid if r["method"] == "Naive_OLS"), None)
    methods = [_nice_name(r["method"]) for r in valid]

    fig = go.Figure()

    for i, r in enumerate(valid):
        color = _method_color(r["method"])
        pct = ((r["se"] / naive_se - 1) * 100) if naive_se and naive_se > 0 else 0

        # Stem line from Naive SE to this method's SE
        if naive_se:
            fig.add_trace(go.Scatter(
                x=[naive_se, r["se"]], y=[i, i],
                mode="lines",
                line=dict(color=color, width=3),
                showlegend=False, hoverinfo="skip",
            ))

        # Dot for this method's SE
        pct_str = (f" ({pct:+.1f}%)" if abs(pct) < 1000
                   else f" ({pct:+.0e}%)") if naive_se else ""
        fig.add_trace(go.Scatter(
            x=[r["se"]], y=[i],
            mode="markers",
            marker=dict(size=14, color=color,
                        line=dict(color="white", width=2)),
            showlegend=False,
            hovertemplate=(
                f"<b>{_nice_name(r['method'])}</b><br>"
                f"SE = {_sfmt(r['se'])}{pct_str}<extra></extra>"
            ),
        ))

        # Label to the right
        fig.add_annotation(
            x=r["se"], y=i,
            text=f"  {_sfmt(r['se'])}{pct_str}",
            showarrow=False, xanchor="left",
            font=dict(size=11, color=color, family="monospace"),
        )

    # Vertical reference at Naive SE
    if naive_se:
        fig.add_shape(
            type="line", x0=naive_se, x1=naive_se,
            y0=-0.5, y1=len(valid) - 0.5,
            line=dict(color="#7F8C8D", width=2, dash="dot"),
        )
        fig.add_annotation(
            x=naive_se, y=len(valid) - 0.5,
            text="Naive SE", showarrow=False, yshift=14,
            font=dict(size=10, color="#7F8C8D"),
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Standard Error",
        yaxis=dict(
            tickvals=list(range(len(valid))),
            ticktext=methods,
        ),
        template=TEMPLATE_LIGHT,
        height=max(300, 50 * len(valid) + 80),
        margin=dict(l=130, r=160, t=50, b=50),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# 12. CI Width Comparison — Horizontal bar chart sorted by width
# ---------------------------------------------------------------------------
def build_ci_bounds_comparison(results_list, theta_true=None,
                                title="Confidence Interval Width"):
    """
    Horizontal bar chart of CI width per method, sorted narrowest-first.
    Color-coded per method for instant recognition.
    """
    valid = [r for r in results_list
             if r["ci_low"] is not None and np.isfinite(r["width"])]
    if not valid:
        return go.Figure().update_layout(title="No data")

    # Sort by width (narrowest at top)
    valid = sorted(valid, key=lambda r: r["width"])
    narrowest = valid[0]["width"]

    fig = go.Figure()

    for i, r in enumerate(valid):
        color = _method_color(r["method"])
        pct_wider = ((r["width"] / narrowest - 1) * 100) if narrowest > 0 else 0
        if i == 0:
            label_suffix = " (narrowest)"
        elif pct_wider < 1000:
            label_suffix = f" (+{pct_wider:.1f}%)"
        else:
            label_suffix = f" (+{pct_wider:.0e}%)"

        fig.add_trace(go.Bar(
            y=[_nice_name(r["method"])],
            x=[r["width"]],
            orientation="h",
            marker=dict(color=color, opacity=0.85,
                        line=dict(color=color, width=1)),
            showlegend=False,
            hovertemplate=(
                f"<b>{_nice_name(r['method'])}</b><br>"
                f"Width = {_sfmt(r['width'])}<br>"
                f"CI = [{_sfmt(r['ci_low'])}, {_sfmt(r['ci_high'])}]"
                f"<extra></extra>"
            ),
        ))

        # Width label to the right
        fig.add_annotation(
            x=r["width"], y=i,
            text=f"  {_sfmt(r['width'])}{label_suffix}",
            showarrow=False, xanchor="left",
            font=dict(size=10, color=color, family="monospace"),
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="CI Width (upper \u2212 lower)",
        yaxis=dict(categoryorder="array",
                   categoryarray=[_nice_name(r["method"]) for r in valid]),
        template=TEMPLATE_LIGHT,
        height=max(300, 45 * len(valid) + 80),
        margin=dict(l=130, r=180, t=50, b=50),
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.3,
    )
    return fig
