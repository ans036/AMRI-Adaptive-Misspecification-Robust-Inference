"""
Interactive Simulation Runner
==============================
Users configure a data source (synthetic DGP, real dataset, or custom CSV),
select inference methods, and compare their performance side by side.
"""

import json
import io
import base64
import dash
from dash import html, dcc, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
import numpy as np

from components.plots import (
    COLORS as METHOD_COLORS,
    build_forest_plot,
    build_scatter_with_fit,
    build_diagnostic_panel,
    build_coverage_bar_chart,
    build_width_bar_chart,
    build_weight_histogram,
    build_accuracy_ranking,
    build_se_comparison,
    build_ci_bounds_comparison,
)
from engine.simulation_engine import (
    DGP_INFO,
    ALL_METHODS,
    REAL_DATASETS,
    ERROR_DISTRIBUTIONS,
    MISSPEC_TYPES,
    run_single_dataset,
    run_on_data,
    load_real_dataset,
    run_monte_carlo,
    custom_dgp,
)


def _fmt(x, digits=4):
    """Smart number formatter: avoids overflow for very large/small values."""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "N/A"
    abs_x = abs(x)
    if abs_x == 0:
        return "0"
    if abs_x >= 1e6:
        return f"{x:.2e}"
    if abs_x >= 100:
        return f"{x:.1f}"
    if abs_x >= 1:
        return f"{x:.{digits}f}"
    # Count leading zeros after decimal
    if abs_x < 1e-6:
        return f"{x:.2e}"
    return f"{x:.{digits}f}"

dash.register_page(__name__, path="/simulation", name="Simulation",
                   title="AMRI — Simulation Runner")

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout(**kwargs):
    return dbc.Container([
        html.H3("Interactive Simulation Runner", className="mb-2"),
        html.P(
            "Choose a data source, select inference methods, and compare "
            "how each method constructs confidence intervals on the same data. "
            "Methods share the same OLS point estimate — the key differences "
            "are in standard errors, CI width, and coverage.",
            className="text-muted mb-4",
        ),

        dbc.Row([
            # ---- LEFT SIDEBAR ----
            dbc.Col([
                # Data Source Card
                dbc.Card([
                    dbc.CardHeader(html.H5("Data Source", className="mb-0")),
                    dbc.CardBody([
                        dbc.Tabs([
                            # Tab 1: Synthetic DGP
                            dbc.Tab([
                                html.Div([
                                    # --- Section 1: Quick presets OR Custom ---
                                    dbc.Label("Scenario Preset",
                                              className="form-label mt-2"),
                                    dbc.Select(
                                        id="sim-dgp",
                                        options=[
                                            {"label": "Custom DGP (full control)",
                                             "value": "Custom_DGP"},
                                            {"label": "-" * 30,
                                             "value": "_sep", "disabled": True},
                                        ] + [
                                            {"label": v["label"], "value": k}
                                            for k, v in DGP_INFO.items()
                                            if k != "Custom_DGP"
                                        ],
                                        value="Custom_DGP",
                                        className="mb-2",
                                    ),
                                    html.Small(id="sim-dgp-desc",
                                               className="text-muted d-block mb-2"),

                                    # Preset-mode note (hidden when Custom)
                                    html.Div(id="sim-preset-note", children=[
                                        dbc.Alert([
                                            html.Small([
                                                "This preset uses built-in settings. ",
                                                "Select ",
                                                html.Strong("Custom DGP"),
                                                " above to choose the error ",
                                                "distribution and parameters.",
                                            ]),
                                        ], color="info", className="py-1 px-2 mb-2"),
                                    ], style={"display": "none"}),

                                    # --- Section 2: Error Distribution ---
                                    # (always visible container, contents
                                    #  toggled between enabled/disabled)
                                    html.Div(id="sim-custom-dgp-opts", children=[
                                        html.Hr(className="my-2"),
                                        html.Div([
                                            html.Span(
                                                "Error Distribution",
                                                style={"fontWeight": "700",
                                                       "fontSize": "0.85rem"}),
                                        ], className="mb-2"),
                                        dbc.Select(
                                            id="sim-error-dist",
                                            options=[
                                                {"label": v["label"], "value": k}
                                                for k, v in ERROR_DISTRIBUTIONS.items()
                                            ],
                                            value="normal", className="mb-2",
                                        ),
                                        # Distribution-specific params
                                        html.Div(id="sim-dist-params", children=[
                                            dbc.Label("Degrees of Freedom (df)",
                                                      className="form-label",
                                                      id="sim-df-label"),
                                            dbc.Input(id="sim-df", type="number",
                                                      value=3, min=2.1, step=0.5,
                                                      className="mb-2"),
                                            dbc.Label("Contamination Prob (p)",
                                                      className="form-label",
                                                      id="sim-contam-p-label"),
                                            dbc.Input(id="sim-contam-p", type="number",
                                                      value=0.1, min=0, max=0.5,
                                                      step=0.05, className="mb-2"),
                                            dbc.Label("Contamination Scale",
                                                      className="form-label",
                                                      id="sim-contam-scale-label"),
                                            dbc.Input(id="sim-contam-scale",
                                                      type="number", value=5,
                                                      min=1, step=1,
                                                      className="mb-2"),
                                            dbc.Label("Skewness Parameter",
                                                      className="form-label",
                                                      id="sim-skew-label"),
                                            dbc.Input(id="sim-skew", type="number",
                                                      value=5, step=1,
                                                      className="mb-2"),
                                            dbc.Label("Gamma Shape (k)",
                                                      className="form-label",
                                                      id="sim-gamma-shape-label"),
                                            dbc.Input(id="sim-gamma-shape",
                                                      type="number", value=2,
                                                      min=0.5, step=0.5,
                                                      className="mb-2"),
                                            dbc.Label("Poisson \u03bb (rate)",
                                                      className="form-label",
                                                      id="sim-poisson-lam-label"),
                                            dbc.Input(id="sim-poisson-lam",
                                                      type="number", value=5,
                                                      min=1, step=1,
                                                      className="mb-2"),
                                            dbc.Label("Chi\u00b2 / Weibull df/shape",
                                                      className="form-label",
                                                      id="sim-chi2-df-label"),
                                            dbc.Input(id="sim-chi2-df",
                                                      type="number", value=3,
                                                      min=1, step=1,
                                                      className="mb-2"),
                                            dbc.Label("Beta \u03b1=\u03b2 / Pareto \u03b1",
                                                      className="form-label",
                                                      id="sim-beta-alpha-label"),
                                            dbc.Input(id="sim-beta-alpha",
                                                      type="number", value=2,
                                                      min=0.5, step=0.5,
                                                      className="mb-2"),
                                        ]),

                                        html.Hr(className="my-2"),
                                        html.Div([
                                            html.Span(
                                                "Misspecification & Model",
                                                style={"fontWeight": "700",
                                                       "fontSize": "0.85rem"}),
                                        ], className="mb-2"),
                                        dbc.Label("Misspecification Type",
                                                  className="form-label"),
                                        dbc.Select(
                                            id="sim-misspec-type",
                                            options=[
                                                {"label": v["label"], "value": k}
                                                for k, v in MISSPEC_TYPES.items()
                                            ],
                                            value="heteroscedastic",
                                            className="mb-2",
                                        ),
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Label("\u03b2\u2080",
                                                          className="form-label"),
                                                dbc.Input(id="sim-beta0",
                                                          type="number",
                                                          value=0, step=0.1,
                                                          className="mb-2"),
                                            ], width=4),
                                            dbc.Col([
                                                dbc.Label("\u03b2\u2081",
                                                          className="form-label"),
                                                dbc.Input(id="sim-beta1",
                                                          type="number",
                                                          value=1, step=0.1,
                                                          className="mb-2"),
                                            ], width=4),
                                            dbc.Col([
                                                dbc.Label("\u03c3",
                                                          className="form-label"),
                                                dbc.Input(id="sim-sigma",
                                                          type="number",
                                                          value=1, min=0.01,
                                                          step=0.1,
                                                          className="mb-2"),
                                            ], width=4),
                                        ]),
                                    ]),

                                    html.Hr(className="my-2"),

                                    dbc.Label("Misspecification (\u03b4)",
                                              className="form-label"),
                                    dcc.Slider(
                                        id="sim-delta", min=0, max=1, step=0.05,
                                        value=0.5,
                                        marks={0: "0", 0.25: ".25", 0.5: ".5",
                                               0.75: ".75", 1: "1"},
                                        tooltip={"placement": "bottom",
                                                 "always_visible": True},
                                        className="mb-3",
                                    ),

                                    dbc.Label("Sample Size (n)", className="form-label"),
                                    dbc.Select(
                                        id="sim-n",
                                        options=[{"label": str(n), "value": n}
                                                 for n in [50, 100, 200, 500,
                                                           1000, 2000]],
                                        value="200",
                                        className="mb-2",
                                    ),

                                    dbc.Label("Random Seed", className="form-label"),
                                    dbc.Input(id="sim-seed", type="number",
                                              value=42, className="mb-2"),
                                ]),
                            ], label="Synthetic DGP", tab_id="tab-synthetic"),

                            # Tab 2: Real Dataset
                            dbc.Tab([
                                html.Div([
                                    dbc.Label("Choose Dataset",
                                              className="form-label mt-2"),
                                    dbc.Select(
                                        id="sim-real-dataset",
                                        options=[
                                            {"label": v["label"], "value": k}
                                            for k, v in REAL_DATASETS.items()
                                        ],
                                        value="California_MedInc",
                                        className="mb-2",
                                    ),
                                    html.Small(id="sim-real-desc",
                                               className="text-muted d-block mb-2"),
                                    dbc.Alert(
                                        "Real data has no known true theta. "
                                        "Coverage cannot be assessed — focus on "
                                        "SE differences and CI width.",
                                        color="info", className="py-2 small",
                                    ),
                                ]),
                            ], label="Real Data", tab_id="tab-real"),

                            # Tab 3: Custom Data
                            dbc.Tab([
                                html.Div([
                                    dbc.Label("Paste X values (comma-separated)",
                                              className="form-label mt-2"),
                                    dbc.Textarea(
                                        id="sim-custom-x",
                                        placeholder="1.2, 3.4, 5.6, 7.8, ...",
                                        className="mb-2", style={"height": "60px",
                                                                  "fontSize": "12px"},
                                    ),
                                    dbc.Label("Paste Y values (comma-separated)",
                                              className="form-label"),
                                    dbc.Textarea(
                                        id="sim-custom-y",
                                        placeholder="2.1, 4.3, 6.5, 8.7, ...",
                                        className="mb-2", style={"height": "60px",
                                                                  "fontSize": "12px"},
                                    ),
                                    dbc.Label("True theta (optional)",
                                              className="form-label"),
                                    dbc.Input(
                                        id="sim-custom-theta", type="number",
                                        placeholder="Leave blank if unknown",
                                        className="mb-2",
                                    ),
                                    html.Small(
                                        "Or upload a CSV with columns X and Y:",
                                        className="text-muted d-block mb-1",
                                    ),
                                    dcc.Upload(
                                        id="sim-upload-csv",
                                        children=dbc.Button(
                                            "Upload CSV", color="secondary",
                                            size="sm", className="w-100",
                                        ),
                                        className="mb-2",
                                    ),
                                    html.Div(id="sim-upload-status",
                                             className="small text-muted"),
                                ]),
                            ], label="Custom", tab_id="tab-custom"),
                        ], id="sim-data-tabs", active_tab="tab-synthetic"),
                    ]),
                ], className="mb-3"),

                # Methods & Settings Card
                dbc.Card([
                    dbc.CardHeader(html.H5("Methods & Settings", className="mb-0")),
                    dbc.CardBody([
                        dbc.Label("Inference Methods", className="form-label"),
                        dbc.Checklist(
                            id="sim-methods",
                            options=[
                                {"label": m.replace("_", " "), "value": m}
                                for m in ALL_METHODS
                            ],
                            value=["Naive_OLS", "Sandwich_HC3", "Sandwich_HC4",
                                   "AKS_Adaptive", "AMRI_v1", "AMRI_v2"],
                            className="method-checklist mb-3",
                        ),

                        dbc.Accordion([
                            dbc.AccordionItem([
                                dbc.Label("Significance Level"),
                                dbc.Select(
                                    id="sim-alpha",
                                    options=[
                                        {"label": "0.01 (99% CI)", "value": "0.01"},
                                        {"label": "0.05 (95% CI)", "value": "0.05"},
                                        {"label": "0.10 (90% CI)", "value": "0.10"},
                                    ],
                                    value="0.05", className="mb-2",
                                ),
                                dbc.Label("AMRI c1 (lower threshold)"),
                                dcc.Slider(
                                    id="sim-c1", min=0.3, max=3.0, step=0.1,
                                    value=1.0, marks={1.0: "1.0"},
                                    tooltip={"placement": "bottom"},
                                    className="mb-2",
                                ),
                                dbc.Label("AMRI c2 (upper threshold)"),
                                dcc.Slider(
                                    id="sim-c2", min=1.0, max=5.0, step=0.1,
                                    value=2.0, marks={2.0: "2.0"},
                                    tooltip={"placement": "bottom"},
                                    className="mb-2",
                                ),
                                dbc.Label("Monte Carlo Replications (B)"),
                                dbc.Select(
                                    id="sim-B",
                                    options=[
                                        {"label": "1 (single run)", "value": "1"},
                                        {"label": "10 (quick MC)", "value": "10"},
                                        {"label": "50", "value": "50"},
                                        {"label": "100", "value": "100"},
                                        {"label": "200", "value": "200"},
                                    ],
                                    value="1", className="mb-2",
                                ),
                            ], title="Advanced Settings"),
                        ], start_collapsed=True, className="mb-3"),

                        dbc.Button(
                            "Run Simulation", id="sim-run-btn",
                            color="primary", size="lg", className="w-100 mb-2",
                        ),
                        dbc.Progress(
                            id="sim-progress", value=0, max=100,
                            striped=True, animated=True, className="mb-2",
                            style={"visibility": "hidden"},
                        ),
                        html.Div(id="sim-status",
                                 className="text-muted small text-center"),
                    ]),
                ]),
            ], md=3, className="sim-sidebar"),

            # ---- RIGHT MAIN AREA ----
            dbc.Col([
                dcc.Store(id="sim-single-store"),
                dcc.Store(id="sim-mc-store"),
                dcc.Store(id="sim-data-store"),

                dbc.Tabs([
                    dbc.Tab([
                        html.Div(id="sim-single-content", children=[
                            dbc.Alert([
                                html.Strong("How to use: "),
                                "1) Choose a data source (synthetic, real, or custom) ",
                                "2) Select methods to compare  ",
                                "3) Click 'Run Simulation'  ",
                                html.Br(), html.Br(),
                                html.Strong("Key insight: "),
                                "All OLS-based methods share the same point estimate. ",
                                "The differences are in standard errors and CI width — ",
                                "set delta > 0 for synthetic data to see how methods "
                                "diverge under misspecification.",
                            ], color="info", className="mt-3"),
                        ]),
                    ], label="Inference Comparison", tab_id="tab-single"),

                    dbc.Tab([
                        html.Div(id="sim-mc-content", children=[
                            dbc.Alert(
                                "Set Monte Carlo replications B > 1 in Advanced "
                                "Settings, then run. Only available with synthetic DGP.",
                                color="info", className="mt-3",
                            ),
                        ]),
                    ], label="Monte Carlo Results", tab_id="tab-mc"),
                ], id="sim-tabs", active_tab="tab-single", className="mt-2"),
            ], md=9),
        ]),
    ], fluid=True)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

# DGP description updater + toggle custom DGP options visibility
@callback(
    Output("sim-dgp-desc", "children"),
    Output("sim-custom-dgp-opts", "style"),
    Output("sim-preset-note", "style"),
    Input("sim-dgp", "value"),
)
def update_dgp_desc(dgp):
    is_custom = (dgp == "Custom_DGP")
    desc = ""
    if dgp and dgp in DGP_INFO:
        info = DGP_INFO[dgp]
        desc = f"{info['desc']}  Formula: {info['formula']}"

    if is_custom:
        # Show all custom controls, hide preset note
        custom_style = {"display": "block"}
        preset_note = {"display": "none"}
    else:
        # Hide custom controls, show preset note
        custom_style = {"display": "none"}
        preset_note = {"display": "block"}

    return desc, custom_style, preset_note


# Toggle distribution-specific parameter visibility
@callback(
    Output("sim-df-label", "style"),
    Output("sim-df", "style"),
    Output("sim-contam-p-label", "style"),
    Output("sim-contam-p", "style"),
    Output("sim-contam-scale-label", "style"),
    Output("sim-contam-scale", "style"),
    Output("sim-skew-label", "style"),
    Output("sim-skew", "style"),
    Output("sim-gamma-shape-label", "style"),
    Output("sim-gamma-shape", "style"),
    Output("sim-poisson-lam-label", "style"),
    Output("sim-poisson-lam", "style"),
    Output("sim-chi2-df-label", "style"),
    Output("sim-chi2-df", "style"),
    Output("sim-beta-alpha-label", "style"),
    Output("sim-beta-alpha", "style"),
    Input("sim-error-dist", "value"),
)
def toggle_dist_params(dist):
    hide = {"display": "none"}
    show = {"display": "block", "marginBottom": "0.5rem"}
    s = {"display": "block"}
    # df: t_heavy
    # contam: contaminated
    # skew: skewed
    # gamma shape: gamma, weibull
    # poisson lam: poisson
    # chi2 df: chi2, weibull
    # beta alpha: beta_sym, pareto
    return (
        show if dist == "t_heavy" else hide,                    # df label
        s if dist == "t_heavy" else hide,                        # df input
        show if dist == "contaminated" else hide,                # contam p
        s if dist == "contaminated" else hide,
        show if dist == "contaminated" else hide,                # contam scale
        s if dist == "contaminated" else hide,
        show if dist == "skewed" else hide,                      # skew
        s if dist == "skewed" else hide,
        show if dist in ("gamma", "weibull") else hide,          # gamma/weibull shape
        s if dist in ("gamma", "weibull") else hide,
        show if dist == "poisson" else hide,                     # poisson lam
        s if dist == "poisson" else hide,
        show if dist in ("chi2", "weibull") else hide,           # chi2 df / weibull
        s if dist in ("chi2", "weibull") else hide,
        show if dist in ("beta_sym", "pareto") else hide,        # beta/pareto alpha
        s if dist in ("beta_sym", "pareto") else hide,
    )


# Real dataset description
@callback(
    Output("sim-real-desc", "children"),
    Input("sim-real-dataset", "value"),
)
def update_real_desc(ds):
    if ds and ds in REAL_DATASETS:
        return REAL_DATASETS[ds]["desc"]
    return ""


# CSV upload handler
@callback(
    Output("sim-custom-x", "value"),
    Output("sim-custom-y", "value"),
    Output("sim-upload-status", "children"),
    Input("sim-upload-csv", "contents"),
    State("sim-upload-csv", "filename"),
    prevent_initial_call=True,
)
def handle_csv_upload(contents, filename):
    if contents is None:
        return no_update, no_update, ""
    try:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        import pandas as pd
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        # Try common column names
        x_col = next((c for c in df.columns if c.lower() in
                       ("x", "predictor", "feature", "x1")), df.columns[0])
        y_col = next((c for c in df.columns if c.lower() in
                       ("y", "response", "target", "outcome")), df.columns[1])
        x_str = ", ".join(f"{v:.4f}" for v in df[x_col].values[:500])
        y_str = ", ".join(f"{v:.4f}" for v in df[y_col].values[:500])
        return x_str, y_str, f"Loaded {filename}: {len(df)} rows, using {x_col} -> {y_col}"
    except Exception as e:
        return no_update, no_update, f"Error: {str(e)}"


# Main simulation callback
@callback(
    Output("sim-single-store", "data"),
    Output("sim-mc-store", "data"),
    Output("sim-data-store", "data"),
    Output("sim-status", "children"),
    Output("sim-progress", "value"),
    Output("sim-progress", "style"),
    Input("sim-run-btn", "n_clicks"),
    State("sim-data-tabs", "active_tab"),
    State("sim-dgp", "value"),
    State("sim-delta", "value"),
    State("sim-n", "value"),
    State("sim-B", "value"),
    State("sim-methods", "value"),
    State("sim-alpha", "value"),
    State("sim-c1", "value"),
    State("sim-c2", "value"),
    State("sim-seed", "value"),
    State("sim-real-dataset", "value"),
    State("sim-custom-x", "value"),
    State("sim-custom-y", "value"),
    State("sim-custom-theta", "value"),
    # Custom DGP parameters
    State("sim-error-dist", "value"),
    State("sim-misspec-type", "value"),
    State("sim-df", "value"),
    State("sim-contam-p", "value"),
    State("sim-contam-scale", "value"),
    State("sim-skew", "value"),
    State("sim-beta0", "value"),
    State("sim-beta1", "value"),
    State("sim-sigma", "value"),
    State("sim-gamma-shape", "value"),
    State("sim-poisson-lam", "value"),
    State("sim-chi2-df", "value"),
    State("sim-beta-alpha", "value"),
    prevent_initial_call=True,
)
def run_simulation(n_clicks, data_tab, dgp, delta, n, B, methods, alpha,
                   c1, c2, seed, real_dataset, custom_x, custom_y,
                   custom_theta, error_dist, misspec_type, df_val,
                   contam_p, contam_scale, skew_val, beta0, beta1, sigma,
                   gamma_shape, poisson_lam, chi2_df, beta_alpha):
    if not n_clicks or not methods:
        return no_update, no_update, no_update, "", 0, {"visibility": "hidden"}

    try:
        alpha = float(alpha) if alpha else 0.05
        c1 = float(c1) if c1 else 1.0
        c2 = float(c2) if c2 else 2.0
        seed = int(seed) if seed else 42
        B = int(B) if B else 1

        # Determine data source
        if data_tab == "tab-real":
            # Real dataset
            X, Y, x_name, y_name = load_real_dataset(real_dataset)
            theta_true = None  # Unknown for real data
            results, se_ratio, intercept, ols_slope = run_on_data(
                X, Y, methods, alpha, c1, c2, theta_true, seed=seed
            )
            single_data = json.dumps({
                "results": results, "theta_true": None,
            })
            data_store = json.dumps({
                "X": X.tolist()[:2000], "Y": Y.tolist()[:2000],
                "theta_true": None, "intercept": intercept,
                "se_ratio": se_ratio, "ols_slope": ols_slope,
                "source": "real", "dataset": real_dataset,
                "x_name": x_name, "y_name": y_name,
                "n": len(X),
            })
            mc_data = None
            status = f"Done! Real data: {real_dataset} (n={len(X)}), {len(methods)} methods"

        elif data_tab == "tab-custom":
            # Custom data
            if not custom_x or not custom_y:
                return (no_update, no_update, no_update,
                        "Error: Paste X and Y values first", 0,
                        {"visibility": "hidden"})
            X = np.array([float(v.strip()) for v in custom_x.split(",")
                          if v.strip()])
            Y = np.array([float(v.strip()) for v in custom_y.split(",")
                          if v.strip()])
            if len(X) != len(Y):
                return (no_update, no_update, no_update,
                        f"Error: X has {len(X)} values but Y has {len(Y)}",
                        0, {"visibility": "hidden"})
            theta_true = float(custom_theta) if custom_theta else None
            results, se_ratio, intercept, ols_slope = run_on_data(
                X, Y, methods, alpha, c1, c2, theta_true, seed=seed
            )
            single_data = json.dumps({
                "results": results, "theta_true": theta_true,
            })
            data_store = json.dumps({
                "X": X.tolist(), "Y": Y.tolist(),
                "theta_true": theta_true, "intercept": intercept,
                "se_ratio": se_ratio, "ols_slope": ols_slope,
                "source": "custom", "n": len(X),
            })
            mc_data = None
            status = f"Done! Custom data (n={len(X)}), {len(methods)} methods"

        else:
            # Synthetic DGP
            n_val = int(n) if n else 200

            if dgp == "Custom_DGP":
                # Custom DGP with user-specified distribution
                error_params = {}
                if error_dist == "t_heavy":
                    error_params["df"] = float(df_val) if df_val else 3
                elif error_dist == "contaminated":
                    error_params["p"] = float(contam_p) if contam_p else 0.1
                    error_params["scale"] = (float(contam_scale)
                                             if contam_scale else 5)
                elif error_dist == "skewed":
                    error_params["skew"] = float(skew_val) if skew_val else 5
                elif error_dist == "gamma":
                    error_params["shape"] = (float(gamma_shape)
                                             if gamma_shape else 2)
                elif error_dist == "poisson":
                    error_params["lam"] = (float(poisson_lam)
                                           if poisson_lam else 5)
                elif error_dist == "chi2":
                    error_params["df"] = (float(chi2_df)
                                          if chi2_df else 3)
                elif error_dist == "beta_sym":
                    error_params["alpha_beta"] = (float(beta_alpha)
                                                  if beta_alpha else 2)
                elif error_dist == "weibull":
                    error_params["shape"] = (float(gamma_shape)
                                             if gamma_shape else 1.5)
                elif error_dist == "pareto":
                    error_params["alpha"] = (float(beta_alpha)
                                             if beta_alpha else 3)
                rng = np.random.default_rng(int(seed) if seed else 42)
                b0 = float(beta0) if beta0 is not None else 0.0
                b1 = float(beta1) if beta1 is not None else 1.0
                sig = float(sigma) if sigma else 1.0
                mtype = misspec_type or "heteroscedastic"

                X_arr, Y_arr, theta_true = custom_dgp(
                    n_val, delta, rng,
                    error_dist=error_dist or "normal",
                    error_params=error_params,
                    misspec_type=mtype,
                    beta0=b0, beta1=b1, sigma=sig,
                )
                results, se_ratio, intercept, ols_slope = run_on_data(
                    X_arr, Y_arr, methods, alpha, c1, c2,
                    theta_true, seed=int(seed) if seed else 42,
                )
                dgp_label = (f"Custom ({error_dist}, "
                             f"{mtype}, \u03b4={delta})")
                X = X_arr.tolist()
                Y = Y_arr.tolist()
            else:
                results, X, Y, theta_true, intercept, se_ratio = \
                    run_single_dataset(
                        dgp, delta, n_val, methods, alpha, c1, c2, seed
                    )
                dgp_label = dgp

            ols_slope = next((r["theta_hat"] for r in results
                              if r["theta_hat"] is not None),
                             theta_true if 'theta_true' in dir() else 1.0)
            single_data = json.dumps({
                "results": results, "theta_true": theta_true,
            })
            data_store = json.dumps({
                "X": X, "Y": Y,
                "theta_true": theta_true, "intercept": intercept,
                "se_ratio": se_ratio, "ols_slope": ols_slope,
                "source": "synthetic", "dgp": dgp_label, "delta": delta,
                "n": n_val,
            })

            mc_data = None
            if B > 1 and dgp != "Custom_DGP":
                mc_summary, all_weights = run_monte_carlo(
                    dgp, delta, n_val, methods, B, alpha, c1, c2, seed
                )
                mc_data = json.dumps({
                    "summary": mc_summary,
                    "weights": all_weights,
                    "alpha": alpha,
                })
            elif B > 1 and dgp == "Custom_DGP":
                # MC for custom DGP: run B replications manually
                from engine.simulation_engine import custom_dgp as _cdgp
                mc_method_data = {m: {"thetas": [], "covers": [],
                                      "widths": [], "ses": []}
                                  for m in methods}
                mc_weights = []
                base_rng = np.random.default_rng(int(seed) if seed else 42)
                for rep in range(B):
                    rep_seed = base_rng.integers(0, 2**31)
                    rep_rng = np.random.default_rng(rep_seed)
                    Xr, Yr, tt = _cdgp(
                        n_val, delta, rep_rng,
                        error_dist=error_dist or "normal",
                        error_params=error_params,
                        misspec_type=mtype,
                        beta0=b0, beta1=b1, sigma=sig,
                    )
                    res_r, _, _, _ = run_on_data(
                        Xr, Yr, methods, alpha, c1, c2, tt,
                        seed=rep_seed,
                    )
                    for r in res_r:
                        m = r["method"]
                        if r["theta_hat"] is not None:
                            mc_method_data[m]["thetas"].append(
                                r["theta_hat"])
                            mc_method_data[m]["covers"].append(
                                r["covers_true"])
                            mc_method_data[m]["widths"].append(r["width"])
                            mc_method_data[m]["ses"].append(r["se"])
                        if m == "AMRI_v2" and r["w"] is not None:
                            mc_weights.append(r["w"])

                mc_summary = []
                nominal = 1 - alpha
                for m in methods:
                    d = mc_method_data[m]
                    valid = len(d["thetas"])
                    if valid == 0:
                        mc_summary.append({
                            "method": m, "coverage": None,
                            "avg_width": None, "bias": None,
                            "rmse": None, "coverage_accuracy": None,
                            "valid_reps": 0, "coverage_mc_se": 0,
                        })
                        continue
                    thetas = np.array(d["thetas"])
                    covers = np.array(d["covers"])
                    widths = np.array(d["widths"])
                    cov = float(np.nanmean(covers))
                    mc_summary.append({
                        "method": m,
                        "coverage": cov,
                        "coverage_mc_se": float(
                            np.sqrt(cov * (1 - cov) / valid)),
                        "avg_width": float(np.nanmean(widths)),
                        "bias": float(np.mean(thetas) - theta_true),
                        "rmse": float(np.sqrt(
                            np.mean((thetas - theta_true) ** 2))),
                        "coverage_accuracy": float(abs(cov - nominal)),
                        "valid_reps": valid,
                    })
                mc_data = json.dumps({
                    "summary": mc_summary,
                    "weights": mc_weights,
                    "alpha": alpha,
                })

            status = (f"Done! {dgp_label}, \u03b4={delta}, n={n_val}, "
                      f"B={B}, {len(methods)} methods")

        return (single_data, mc_data, data_store, status, 100,
                {"visibility": "hidden"})

    except Exception as e:
        return (no_update, no_update, no_update,
                f"Error: {str(e)}", 0, {"visibility": "hidden"})


# ---------------------------------------------------------------------------
# Render single-run results (THE MAIN VISUALIZATION)
# ---------------------------------------------------------------------------
@callback(
    Output("sim-single-content", "children"),
    Input("sim-single-store", "data"),
    Input("sim-data-store", "data"),
    State("sim-c1", "value"),
    State("sim-c2", "value"),
)
def render_single_run(single_json, data_json, c1, c2):
    if not single_json:
        return dbc.Alert("Run a simulation to see results.", color="info")

    single = json.loads(single_json)
    data = json.loads(data_json)
    results = single["results"]
    theta_true = single["theta_true"]
    X = data["X"]
    Y = data["Y"]
    n = data["n"]
    intercept = data["intercept"]
    se_ratio = data.get("se_ratio", 1.0)
    source = data.get("source", "synthetic")

    # Build title
    if source == "synthetic":
        title_prefix = f"{data.get('dgp', '')} (\u03b4={data.get('delta', '')}, n={n})"
    elif source == "real":
        title_prefix = f"Real: {data.get('dataset', '').replace('_', ' ')} (n={n})"
    else:
        title_prefix = f"Custom Data (n={n})"

    # AMRI v2 result for diagnostics
    amri_result = next((r for r in results if r["method"] == "AMRI_v2"), None)

    # Find Naive SE as baseline for comparison
    naive_result = next((r for r in results if r["method"] == "Naive_OLS"), None)
    naive_se = naive_result["se"] if naive_result and naive_result["se"] else None

    # ---- Summary info banner ----
    valid_results = [r for r in results if r["theta_hat"] is not None]
    # Filter out extreme SE values (Inf/NaN) for stats
    ses = [r["se"] for r in valid_results
           if r["se"] is not None and np.isfinite(r["se"])]
    se_spread = ((max(ses) / min(ses) - 1) * 100
                 if ses and min(ses) > 0 else 0)

    # Misspecification severity indicator
    if se_spread > 10:
        severity = ("High", "#C0392B")
    elif se_spread > 3:
        severity = ("Moderate", "#E67E22")
    else:
        severity = ("Low", "#27AE60")

    # Format SE ratio safely
    se_ratio_safe = se_ratio if np.isfinite(se_ratio) else 0
    spread_str = (f"{se_spread:.1f}%" if se_spread < 1e4
                  else f"{se_spread:.0e}%")

    info_items = [
        ("SE Ratio (HC3/Naive)", _fmt(se_ratio_safe, 3),
         "#C0392B" if abs(se_ratio_safe - 1) > 0.1 else "#27AE60"),
        ("SE Spread", spread_str, severity[1]),
        ("Misspecification Signal", severity[0], severity[1]),
    ]
    if amri_result and amri_result.get("w") is not None:
        w_val = amri_result["w"]
        w_color = ("#1A5276" if w_val < 0.3
                   else ("#E67E22" if w_val < 0.7 else "#C0392B"))
        info_items.append(("AMRI Weight", f"{w_val:.3f}", w_color))

    info_banner = html.Div(
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.Div(label, style={"fontSize": "0.7rem",
                                            "textTransform": "uppercase",
                                            "letterSpacing": "0.05em",
                                            "color": "#888"}),
                    html.Div(val, style={"fontSize": "1.3rem",
                                          "fontWeight": "700",
                                          "color": color,
                                          "fontFamily": "monospace",
                                          "overflow": "hidden",
                                          "textOverflow": "ellipsis",
                                          "whiteSpace": "nowrap"}),
                ], style={"textAlign": "center", "padding": "8px 0"}),
                md=True,
            )
            for label, val, color in info_items
        ]),
        style={
            "background": "#f8f9fa",
            "borderRadius": "8px",
            "padding": "4px 12px",
            "marginBottom": "16px",
            "border": "1px solid #eee",
        },
    )

    # ---- Method comparison cards (side by side) ----
    # Sort: AMRI methods first, then by SE
    def _card_sort(r):
        is_amri = "AMRI" in r["method"]
        return (0 if is_amri else 1, r["se"] or 999)

    sorted_results = sorted(valid_results, key=_card_sort)

    method_cards = []
    for r in sorted_results:
        is_amri = "AMRI" in r["method"]
        m_color = METHOD_COLORS.get(r["method"], "#666")
        se_val = r["se"] if r["se"] is not None and np.isfinite(r["se"]) else None

        # SE relative to Naive — always show the actual %
        se_pct = ""
        se_pct_color = "#666"
        if (naive_se and naive_se > 0 and se_val is not None):
            pct = (se_val / naive_se - 1) * 100
            if abs(pct) < 0.5:
                se_pct = "\u2248 Naive"
                se_pct_color = "#7F8C8D"
            elif pct > 0:
                pct_str = f"{pct:.1f}" if pct < 1000 else f"{pct:.0e}"
                se_pct = f"+{pct_str}%"
                se_pct_color = "#C0392B"
            else:
                se_pct = f"{pct:.1f}%"
                se_pct_color = "#27AE60"

        cover_icon = ""
        cover_color = ""
        if r["covers_true"] is True:
            cover_icon = "\u2713"
            cover_color = "#27AE60"
        elif r["covers_true"] is False:
            cover_icon = "\u2717"
            cover_color = "#C0392B"

        method_cards.append(
            dbc.Col(
                html.Div([
                    # Color bar at top
                    html.Div(style={
                        "height": "4px",
                        "background": m_color,
                        "borderRadius": "4px 4px 0 0",
                    }),
                    html.Div([
                        # Method name
                        html.Div(
                            r["method"].replace("_", " "),
                            style={
                                "fontWeight": "700" if is_amri else "600",
                                "fontSize": "0.85rem",
                                "color": m_color,
                                "marginBottom": "4px",
                            },
                        ),
                        # SE value + % difference
                        html.Div([
                            html.Span(
                                _fmt(se_val) if se_val else "N/A",
                                style={"fontSize": "1.05rem",
                                       "fontWeight": "700",
                                       "fontFamily": "monospace"}),
                            html.Span(
                                f" {se_pct}",
                                style={"fontSize": "0.75rem",
                                       "fontWeight": "600",
                                       "color": se_pct_color})
                            if se_pct else None,
                        ]),
                        # CI
                        html.Div(
                            f"[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]",
                            style={"fontSize": "0.75rem", "color": "#555",
                                   "fontFamily": "monospace",
                                   "marginTop": "3px",
                                   "overflow": "hidden",
                                   "textOverflow": "ellipsis",
                                   "whiteSpace": "nowrap"},
                        ),
                        # Width
                        html.Div(
                            f"W = {_fmt(r['width'])}",
                            style={"fontSize": "0.75rem", "color": "#666",
                                   "fontFamily": "monospace"},
                        ),
                        # Coverage + weight row
                        html.Div([
                            html.Span(
                                cover_icon,
                                style={"color": cover_color,
                                       "fontWeight": "bold",
                                       "marginRight": "4px"},
                            ) if cover_icon else None,
                            html.Span(
                                f"w={r['w']:.2f}",
                                style={"fontSize": "0.72rem",
                                       "color": m_color,
                                       "fontWeight": "600"},
                            ) if r.get("w") is not None else None,
                        ], style={"marginTop": "2px"}),
                    ], style={"padding": "6px 8px"}),
                ], style={
                    "border": f"1px solid {m_color}33",
                    "borderRadius": "6px",
                    "background": f"{m_color}08" if is_amri else "white",
                    "height": "100%",
                    "overflow": "hidden",
                }),
                xs=6, sm=4, md=True, className="mb-2 px-1",
            )
        )

    # ---- Build summary table (emphasize differences, not theta_hat) ----
    theta_hat_val = valid_results[0]["theta_hat"] if valid_results else None

    table_rows = []
    for r in valid_results:
        se_finite = (r["se"] is not None and np.isfinite(r["se"]))
        if naive_se and naive_se > 0 and se_finite:
            ratio_val = r["se"] / naive_se
            ratio_str = _fmt(ratio_val, 3) + "\u00d7"
            if ratio_val > 1.1:
                ratio_cell = html.Td(
                    html.Span(ratio_str, style={"color": "#C0392B",
                                                 "fontWeight": "bold"}))
            elif ratio_val < 0.9:
                ratio_cell = html.Td(
                    html.Span(ratio_str, style={"color": "#27AE60",
                                                 "fontWeight": "bold"}))
            else:
                ratio_cell = html.Td(ratio_str)
        else:
            ratio_cell = html.Td("\u2014")

        covers_badge = None
        if r["covers_true"] is True:
            covers_badge = dbc.Badge("\u2713 Yes", color="success")
        elif r["covers_true"] is False:
            covers_badge = dbc.Badge("\u2717 No", color="danger")
        else:
            covers_badge = html.Span("\u2014", className="text-muted")

        table_rows.append(html.Tr([
            html.Td(html.Strong(r["method"].replace("_", " "))
                     if "AMRI" in r["method"]
                     else r["method"].replace("_", " ")),
            html.Td(html.Strong(_fmt(r["se"]))),
            ratio_cell,
            html.Td(f"[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]"),
            html.Td(html.Strong(_fmt(r["width"]))),
            html.Td(covers_badge),
            html.Td(f"{r['w']:.3f}" if r["w"] is not None else "\u2014"),
        ]))

    # OLS slope display
    ols_slope_val = data.get("ols_slope")
    ols_slope_str = _fmt(ols_slope_val) if ols_slope_val is not None else "N/A"

    return html.Div([
        # Info banner
        info_banner,

        # Row 0: Method comparison cards (side by side)
        html.H6("Method Comparison (Side by Side)", className="mb-2"),
        dbc.Row(method_cards, className="mb-3 g-2"),

        # Row 1: Forest plot
        dbc.Card([
            dbc.CardBody(
                dcc.Graph(
                    figure=build_forest_plot(
                        results, theta_true,
                        title=f"Inference Results \u2014 {title_prefix}"
                    ),
                    config={"displayModeBar": True},
                )
            ),
        ], className="mb-3"),

        # Row 2: SE comparison + CI bounds
        dbc.Row([
            dbc.Col(
                dbc.Card(dbc.CardBody(
                    dcc.Graph(figure=build_se_comparison(
                        results,
                        title="SE Comparison (Key Differences)"
                    ))
                )),
                md=6,
            ),
            dbc.Col(
                dbc.Card(dbc.CardBody(
                    dcc.Graph(figure=build_ci_bounds_comparison(
                        results, theta_true,
                        title="CI Bounds & Width"
                    ))
                )),
                md=6,
            ),
        ], className="mb-3 g-3"),

        # Row 3: Scatter + Diagnostic
        dbc.Row([
            dbc.Col(
                dbc.Card(dbc.CardBody(
                    dcc.Graph(figure=build_scatter_with_fit(
                        X, Y,
                        theta_true if theta_true else data.get("ols_slope", 1.0),
                        intercept,
                        title=f"Data (n={n})"
                    ))
                )),
                md=7,
            ),
            dbc.Col(
                dbc.Card(dbc.CardBody([
                    dcc.Graph(figure=build_diagnostic_panel(
                        se_ratio,
                        amri_result["w"] if amri_result and
                        amri_result["w"] is not None else 0.0,
                        float(c1), float(c2), n,
                    )) if amri_result else html.P(
                        "Select AMRI v2 to see diagnostics"),
                    html.Hr(),
                    html.Div([
                        html.Strong("Data Source: "),
                        html.Span(source.capitalize()),
                        html.Br(),
                        html.Strong("True \u03b8: "),
                        html.Span(_fmt(theta_true)
                                  if theta_true is not None else "Unknown"),
                        html.Br(),
                        html.Strong("OLS slope (\u03b8\u0302): "),
                        html.Span(ols_slope_str),
                        html.Br(),
                        html.Strong("n: "), html.Span(f"{n}"),
                    ], className="small"),
                ])),
                md=5,
            ),
        ], className="mb-3 g-3"),

        # Row 4: Summary table
        dbc.Card([
            dbc.CardHeader([
                html.H6([
                    "Full Inference Summary",
                    html.Small(
                        f"  \u2014  \u03b8\u0302 = {_fmt(theta_hat_val)} "
                        "(shared across all OLS-based methods)"
                        if theta_hat_val is not None else "",
                        className="text-muted",
                    ),
                ], className="mb-0"),
            ]),
            dbc.CardBody(
                dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("Method"),
                        html.Th("SE", title="Standard Error"),
                        html.Th("SE / Naive",
                                title="SE relative to Naive OLS"),
                        html.Th("95% CI"),
                        html.Th("Width",
                                title="CI upper \u2212 CI lower"),
                        html.Th("Covers \u03b8?"),
                        html.Th("w", title="AMRI blending weight"),
                    ])),
                    html.Tbody(table_rows),
                ], bordered=True, hover=True, responsive=True, size="sm"),
            ),
        ], className="mb-3"),
    ])


# ---------------------------------------------------------------------------
# Render Monte Carlo results
# ---------------------------------------------------------------------------
@callback(
    Output("sim-mc-content", "children"),
    Input("sim-mc-store", "data"),
)
def render_mc_results(mc_json):
    if not mc_json:
        return dbc.Alert(
            "Set Monte Carlo replications B > 1 in Advanced Settings, then run. "
            "Only available with synthetic DGP data.",
            color="info",
        )

    mc = json.loads(mc_json)
    summary = mc["summary"]
    weights = mc["weights"]
    alpha = mc.get("alpha", 0.05)

    return html.Div([
        # Row 1: Coverage + Width
        dbc.Row([
            dbc.Col(
                dbc.Card(dbc.CardBody(
                    dcc.Graph(figure=build_coverage_bar_chart(summary, alpha=alpha))
                )),
                md=6,
            ),
            dbc.Col(
                dbc.Card(dbc.CardBody(
                    dcc.Graph(figure=build_width_bar_chart(summary))
                )),
                md=6,
            ),
        ], className="mb-3 g-3"),

        # Row 2: Accuracy + Weights
        dbc.Row([
            dbc.Col(
                dbc.Card(dbc.CardBody(
                    dcc.Graph(figure=build_accuracy_ranking(summary, alpha=alpha))
                )),
                md=6,
            ),
            dbc.Col(
                dbc.Card(dbc.CardBody(
                    dcc.Graph(figure=build_weight_histogram(weights))
                )) if weights else dbc.Card(dbc.CardBody(
                    html.P("Select AMRI v2 to see weight distribution",
                           className="text-muted")
                )),
                md=6,
            ),
        ], className="mb-3 g-3"),

        # Row 3: Full table
        dbc.Card([
            dbc.CardHeader(html.H6("Monte Carlo Summary", className="mb-0")),
            dbc.CardBody(
                dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("Method"), html.Th("Coverage"),
                        html.Th("MC SE"), html.Th("Avg Width"),
                        html.Th("Bias"), html.Th("RMSE"),
                        html.Th("|Cov - Nom|"), html.Th("Reps"),
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td(html.Strong(r["method"].replace("_", " "))
                                     if "AMRI" in r["method"]
                                     else r["method"].replace("_", " ")),
                            html.Td(f"{r['coverage']:.4f}"
                                     if r["coverage"] else "—"),
                            html.Td(f"{r.get('coverage_mc_se', 0):.4f}"),
                            html.Td(f"{r['avg_width']:.4f}"
                                     if r["avg_width"] else "—"),
                            html.Td(f"{r['bias']:.4f}"
                                     if r["bias"] is not None else "—"),
                            html.Td(f"{r['rmse']:.4f}"
                                     if r["rmse"] is not None else "—"),
                            html.Td(f"{r['coverage_accuracy']:.4f}"
                                     if r.get("coverage_accuracy") else "—"),
                            html.Td(str(r["valid_reps"])),
                        ]) for r in summary
                    ]),
                ], bordered=True, hover=True, responsive=True, size="sm"),
            ),
        ]),
    ])
