"""
Interactive Simulation Runner
==============================
THE MAIN PAGE. Users configure DGP, parameters, and methods,
then run simulations and see inference results for all methods.
"""

import json
import dash
from dash import html, dcc, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc

from components.plots import (
    build_forest_plot,
    build_scatter_with_fit,
    build_diagnostic_panel,
    build_coverage_bar_chart,
    build_width_bar_chart,
    build_weight_histogram,
    build_accuracy_ranking,
)
from engine.simulation_engine import (
    DGP_INFO,
    ALL_METHODS,
    run_single_dataset,
    run_monte_carlo,
)

dash.register_page(__name__, path="/simulation", name="Simulation",
                   title="AMRI — Simulation Runner")

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout(**kwargs):
    return dbc.Container([
        html.H3("Interactive Simulation Runner", className="mb-3"),
        html.P(
            "Configure a data-generating process and run all inference methods "
            "on the same dataset. See how each method's confidence interval "
            "compares to the true parameter.",
            className="text-muted mb-4",
        ),

        dbc.Row([
            # ---- LEFT SIDEBAR: Configuration ----
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Configuration", className="mb-0")),
                    dbc.CardBody([
                        # DGP selector
                        dbc.Label("Data Generating Process", className="form-label"),
                        dbc.Select(
                            id="sim-dgp",
                            options=[
                                {"label": f"{k}: {v['label']}", "value": k}
                                for k, v in DGP_INFO.items()
                            ],
                            value="Heteroscedastic",
                            className="mb-3",
                        ),
                        html.Small(id="sim-dgp-desc", className="text-muted d-block mb-3"),

                        # Delta slider
                        dbc.Label("Misspecification Severity (delta)", className="form-label"),
                        dcc.Slider(
                            id="sim-delta", min=0, max=1, step=0.05, value=0.5,
                            marks={i / 4: f"{i / 4:.2f}" for i in range(5)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className="mb-3",
                        ),

                        # Sample size
                        dbc.Label("Sample Size (n)", className="form-label"),
                        dbc.Select(
                            id="sim-n",
                            options=[{"label": str(n), "value": n}
                                     for n in [50, 100, 250, 500, 1000, 2000]],
                            value="200",
                            className="mb-3",
                        ),

                        # Replications
                        dbc.Label("Replications (B)", className="form-label"),
                        dbc.Select(
                            id="sim-B",
                            options=[
                                {"label": "1 (single run only)", "value": "1"},
                                {"label": "10 (quick)", "value": "10"},
                                {"label": "50", "value": "50"},
                                {"label": "100", "value": "100"},
                                {"label": "200", "value": "200"},
                                {"label": "500 (thorough)", "value": "500"},
                            ],
                            value="1",
                            className="mb-3",
                        ),

                        # Methods checklist
                        dbc.Label("Methods", className="form-label"),
                        dbc.Checklist(
                            id="sim-methods",
                            options=[
                                {"label": m.replace("_", " "), "value": m}
                                for m in ALL_METHODS
                            ],
                            value=["Naive_OLS", "Sandwich_HC3", "AKS_Adaptive",
                                   "AMRI_v1", "AMRI_v2"],
                            className="method-checklist mb-3",
                        ),

                        # Advanced settings (collapsed)
                        dbc.Accordion([
                            dbc.AccordionItem([
                                dbc.Label("Significance Level (alpha)"),
                                dbc.Select(
                                    id="sim-alpha",
                                    options=[
                                        {"label": "0.01 (99% CI)", "value": "0.01"},
                                        {"label": "0.05 (95% CI)", "value": "0.05"},
                                        {"label": "0.10 (90% CI)", "value": "0.10"},
                                    ],
                                    value="0.05",
                                    className="mb-2",
                                ),
                                dbc.Label("AMRI c1 (lower threshold)"),
                                dcc.Slider(
                                    id="sim-c1", min=0.3, max=3.0, step=0.1,
                                    value=1.0, marks={1.0: "1.0 (default)"},
                                    tooltip={"placement": "bottom"},
                                    className="mb-2",
                                ),
                                dbc.Label("AMRI c2 (upper threshold)"),
                                dcc.Slider(
                                    id="sim-c2", min=1.0, max=5.0, step=0.1,
                                    value=2.0, marks={2.0: "2.0 (default)"},
                                    tooltip={"placement": "bottom"},
                                    className="mb-2",
                                ),
                                dbc.Label("Random Seed"),
                                dbc.Input(
                                    id="sim-seed", type="number", value=42,
                                    className="mb-2",
                                ),
                            ], title="Advanced Settings"),
                        ], start_collapsed=True, className="mb-3"),

                        # Run button
                        dbc.Button(
                            "Run Simulation",
                            id="sim-run-btn",
                            color="primary",
                            size="lg",
                            className="w-100 mb-2",
                        ),

                        # Progress bar (hidden by default)
                        dbc.Progress(
                            id="sim-progress",
                            value=0, max=100,
                            striped=True, animated=True,
                            className="mb-2",
                            style={"visibility": "hidden"},
                        ),

                        # Status text
                        html.Div(id="sim-status", className="text-muted small text-center"),
                    ]),
                ]),
            ], md=3, className="sim-sidebar"),

            # ---- RIGHT MAIN AREA: Results ----
            dbc.Col([
                # Data stores (hidden)
                dcc.Store(id="sim-single-store"),
                dcc.Store(id="sim-mc-store"),
                dcc.Store(id="sim-data-store"),

                dbc.Tabs([
                    # Tab 1: Single-Run Inference
                    dbc.Tab([
                        html.Div(id="sim-single-content", children=[
                            dbc.Alert(
                                "Configure parameters on the left and click "
                                "'Run Simulation' to see inference results.",
                                color="info",
                                className="mt-3",
                            ),
                        ]),
                    ], label="Single-Run Inference", tab_id="tab-single"),

                    # Tab 2: Monte Carlo Results
                    dbc.Tab([
                        html.Div(id="sim-mc-content", children=[
                            dbc.Alert(
                                "Set B > 1 and run to see Monte Carlo coverage "
                                "and width results.",
                                color="info",
                                className="mt-3",
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

# DGP description updater
@callback(
    Output("sim-dgp-desc", "children"),
    Input("sim-dgp", "value"),
)
def update_dgp_desc(dgp):
    if dgp and dgp in DGP_INFO:
        info = DGP_INFO[dgp]
        return f"{info['desc']}  Formula: {info['formula']}"
    return ""


# Main simulation callback
@callback(
    Output("sim-single-store", "data"),
    Output("sim-mc-store", "data"),
    Output("sim-data-store", "data"),
    Output("sim-status", "children"),
    Output("sim-progress", "value"),
    Output("sim-progress", "style"),
    Input("sim-run-btn", "n_clicks"),
    State("sim-dgp", "value"),
    State("sim-delta", "value"),
    State("sim-n", "value"),
    State("sim-B", "value"),
    State("sim-methods", "value"),
    State("sim-alpha", "value"),
    State("sim-c1", "value"),
    State("sim-c2", "value"),
    State("sim-seed", "value"),
    prevent_initial_call=True,
)
def run_simulation(n_clicks, dgp, delta, n, B, methods, alpha, c1, c2, seed):
    if not n_clicks or not methods:
        return no_update, no_update, no_update, "", 0, {"visibility": "hidden"}

    n = int(n)
    B = int(B)
    alpha = float(alpha)
    c1 = float(c1)
    c2 = float(c2)
    seed = int(seed) if seed else 42

    # Show progress
    progress_style = {"visibility": "visible"} if B > 1 else {"visibility": "hidden"}

    # 1. Single run (always)
    results, X, Y, theta_true, intercept, se_ratio = run_single_dataset(
        dgp, delta, n, methods, alpha, c1, c2, seed
    )
    single_data = json.dumps({
        "results": results,
        "theta_true": theta_true,
    })
    data_store = json.dumps({
        "X": X, "Y": Y,
        "theta_true": theta_true,
        "intercept": intercept,
        "se_ratio": se_ratio,
        "dgp": dgp, "delta": delta, "n": n,
    })

    # 2. Monte Carlo (if B > 1)
    mc_data = None
    if B > 1:
        mc_summary, all_weights = run_monte_carlo(
            dgp, delta, n, methods, B, alpha, c1, c2, seed
        )
        mc_data = json.dumps({
            "summary": mc_summary,
            "weights": all_weights,
            "alpha": alpha,
        })

    status = f"Done! B={B}, n={n}, delta={delta}, {len(methods)} methods"
    return single_data, mc_data, data_store, status, 100, {"visibility": "hidden"}


# Render single-run results
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

    # Find AMRI v2 result for diagnostic
    amri_result = next((r for r in results if r["method"] == "AMRI_v2"), None)

    # Build summary table
    table_rows = []
    for r in results:
        if r["theta_hat"] is None:
            continue
        covers_badge = dbc.Badge("Yes", color="success") if r["covers_true"] else dbc.Badge("No", color="danger")
        table_rows.append(html.Tr([
            html.Td(r["method"].replace("_", " ")),
            html.Td(f"{r['theta_hat']:.4f}"),
            html.Td(f"{r['se']:.4f}"),
            html.Td(f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"),
            html.Td(f"{r['width']:.4f}"),
            html.Td(covers_badge),
            html.Td(f"{r['w']:.3f}" if r["w"] is not None else "—"),
        ]))

    return html.Div([
        # Row 1: Forest plot
        dbc.Card([
            dbc.CardBody(
                dcc.Graph(
                    figure=build_forest_plot(
                        results, theta_true,
                        title=f"Inference Results — {data['dgp']} (delta={data['delta']}, n={n})"
                    ),
                    config={"displayModeBar": True},
                )
            ),
        ], className="mb-3"),

        # Row 2: Scatter + Diagnostic
        dbc.Row([
            dbc.Col(
                dbc.Card(dbc.CardBody(
                    dcc.Graph(figure=build_scatter_with_fit(
                        X, Y, theta_true, intercept,
                        title=f"Generated Data (n={n})"
                    ))
                )),
                md=7,
            ),
            dbc.Col(
                dbc.Card(dbc.CardBody([
                    dcc.Graph(figure=build_diagnostic_panel(
                        se_ratio,
                        amri_result["w"] if amri_result and amri_result["w"] is not None else 0.0,
                        float(c1), float(c2), n,
                    )) if amri_result else html.P("AMRI v2 not selected"),
                    # Key metrics
                    html.Hr(),
                    html.Div([
                        html.Strong("True theta: "), html.Span(f"{theta_true:.4f}"),
                        html.Br(),
                        html.Strong("n: "), html.Span(f"{n}"),
                        html.Br(),
                        html.Strong("DGP: "), html.Span(data["dgp"]),
                        html.Br(),
                        html.Strong("delta: "), html.Span(f"{data['delta']}"),
                    ], className="small"),
                ])),
                md=5,
            ),
        ], className="mb-3 g-3"),

        # Row 3: Summary table
        dbc.Card([
            dbc.CardHeader(html.H6("Inference Summary", className="mb-0")),
            dbc.CardBody(
                dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("Method"), html.Th("theta_hat"), html.Th("SE"),
                        html.Th("95% CI"), html.Th("Width"),
                        html.Th("Covers True?"), html.Th("Weight"),
                    ])),
                    html.Tbody(table_rows),
                ], bordered=True, hover=True, responsive=True, size="sm"),
            ),
        ], className="mb-3"),
    ])


# Render Monte Carlo results
@callback(
    Output("sim-mc-content", "children"),
    Input("sim-mc-store", "data"),
)
def render_mc_results(mc_json):
    if not mc_json:
        return dbc.Alert(
            "Set replications (B) > 1 and run to see Monte Carlo results.",
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

        # Row 2: Accuracy ranking + Weight histogram
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

        # Row 3: Full results table
        dbc.Card([
            dbc.CardHeader(html.H6("Monte Carlo Summary", className="mb-0")),
            dbc.CardBody(
                dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("Method"), html.Th("Coverage"),
                        html.Th("MC SE"), html.Th("Avg Width"),
                        html.Th("Bias"), html.Th("RMSE"),
                        html.Th("|Cov - Nom|"), html.Th("Valid Reps"),
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td(r["method"].replace("_", " ")),
                            html.Td(f"{r['coverage']:.4f}" if r["coverage"] else "—"),
                            html.Td(f"{r.get('coverage_mc_se', 0):.4f}"),
                            html.Td(f"{r['avg_width']:.4f}" if r["avg_width"] else "—"),
                            html.Td(f"{r['bias']:.4f}" if r["bias"] is not None else "—"),
                            html.Td(f"{r['rmse']:.4f}" if r["rmse"] is not None else "—"),
                            html.Td(f"{r['coverage_accuracy']:.4f}" if r.get("coverage_accuracy") else "—"),
                            html.Td(str(r["valid_reps"])),
                        ]) for r in summary
                    ]),
                ], bordered=True, hover=True, responsive=True, size="sm"),
            ),
        ]),
    ])
