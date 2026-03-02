"""
About / Documentation Page
===========================
Method explanation, algorithm details, DGP descriptions, references.
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/about", name="About",
                   title="AMRI — About")


def layout(**kwargs):
    return dbc.Container([
        html.H3("About AMRI", className="mb-4"),

        dbc.Row([
            dbc.Col([
                # Method explanation
                dbc.Card([
                    dbc.CardHeader(html.H5("The AMRI Method", className="mb-0")),
                    dbc.CardBody(dcc.Markdown("""
### Adaptive Misspecification-Robust Inference

**Problem:** When fitting a regression model, practitioners face a dilemma:
- **Naive (model-based) SEs** are efficient when the model is correct, but
  produce invalid confidence intervals under misspecification.
- **Robust (sandwich) SEs** are always valid asymptotically, but wider than
  necessary when the model is correct.

**Solution:** AMRI v2 adaptively blends the two approaches using a data-driven
weight that responds to the *SE ratio* — a natural diagnostic for misspecification.

### Algorithm (AMRI v2)

```
Input: X, Y, alpha=0.05, c1=1.0, c2=2.0

1. Fit OLS: Y = a + bX + e
2. Compute SE_naive (model-based) and SE_HC3 (sandwich robust)
3. R = SE_HC3 / SE_naive           (SE ratio)
4. s = |log(R)|                     (symmetric, scale-invariant diagnostic)
5. lo = c1 / sqrt(n)               (lower threshold, shrinks with n)
6. hi = c2 / sqrt(n)               (upper threshold)
7. w = clip((s - lo) / (hi - lo), 0, 1)   (blending weight)
8. SE_AMRI = (1-w) * SE_naive + w * SE_HC3
9. CI = theta_hat +/- t_{n-2, 1-alpha/2} * SE_AMRI
```

### Key Properties

1. **Coverage Continuity** (Theorem 1): The map delta -> Coverage is smooth
   (no discontinuous jumps, addressing the Leeb-Potscher pre-test critique)

2. **Asymptotic Validity** (Theorem 2): Coverage >= 1-alpha asymptotically
   for all delta in [0, delta_max]

3. **Near-Oracle Efficiency** (Theorem 3): Under correct specification,
   width overhead is O(1/n) — negligible
                    """)),
                ], className="mb-4"),

                # DGP descriptions
                dbc.Card([
                    dbc.CardHeader(html.H5("Data Generating Processes (DGPs)", className="mb-0")),
                    dbc.CardBody(
                        dbc.Table([
                            html.Thead(html.Tr([
                                html.Th("DGP"), html.Th("Name"),
                                html.Th("Misspecification Type"), html.Th("Formula"),
                            ])),
                            html.Tbody([
                                html.Tr([html.Td("1"), html.Td("Nonlinearity"),
                                         html.Td("Functional form"), html.Td("Y = X + delta*X^2 + e")]),
                                html.Tr([html.Td("2"), html.Td("Heavy Tails"),
                                         html.Td("Distributional"), html.Td("e ~ t(10 - 8*delta)")]),
                                html.Tr([html.Td("3"), html.Td("Heteroscedasticity"),
                                         html.Td("Variance misspecification"), html.Td("Var(e|X) = exp(delta*X)")]),
                                html.Tr([html.Td("4"), html.Td("Omitted Variable"),
                                         html.Td("Bias from omission"), html.Td("Y = X1 + delta*X2 + e")]),
                                html.Tr([html.Td("5"), html.Td("Clustering"),
                                         html.Td("Ignored dependence"), html.Td("Y = X + sqrt(delta)*u_g + e")]),
                                html.Tr([html.Td("6"), html.Td("Contaminated"),
                                         html.Td("Outlier contamination"), html.Td("e ~ (1-p)*N(0,1) + p*N(0,25)")]),
                            ]),
                        ], bordered=True, hover=True, responsive=True, size="sm"),
                    ),
                ], className="mb-4"),
            ], md=8),

            dbc.Col([
                # Method descriptions
                dbc.Card([
                    dbc.CardHeader(html.H5("Inference Methods", className="mb-0")),
                    dbc.CardBody(dcc.Markdown("""
**Naive OLS** — Model-based SE assuming homoscedasticity.

**Sandwich HC3** — MacKinnon & White (1985). Stata's default robust SE.

**HC4** — Cribari-Neto (2004). Leverage-adjusted robust SE.

**HC5** — Cribari-Neto & da Silva (2011). Max-leverage-aware.

**Pairs Bootstrap** — Resample (X,Y) pairs, percentile CI.

**Wild Bootstrap** — Rademacher weights, preserves heteroscedasticity.

**Bootstrap-t** — Studentized bootstrap for refined coverage.

**AKS Adaptive** — Armstrong, Kline & Sun (2025, Econometrica).
Soft-thresholds toward efficient SE. Optimizes estimation regret.

**AMRI v1** — Hard-switching between naive and robust SE.

**AMRI v2** — Soft-threshold blending (recommended). Near-minimax
optimal for CI construction.
                    """)),
                ], className="mb-4"),

                # References
                dbc.Card([
                    dbc.CardHeader(html.H5("Key References", className="mb-0")),
                    dbc.CardBody(dcc.Markdown("""
- Armstrong, Kline & Sun (2025). "Adapting to Misspecification."
  *Econometrica* 93(6): 1981-2005.

- White (1980). "A Heteroskedasticity-Consistent Covariance Matrix
  Estimator." *Econometrica* 48(4).

- MacKinnon & White (1985). "Some Heteroskedasticity-Consistent
  Covariance Matrix Estimators." *JBES* 3(3).

- Leeb & Potscher (2005). "Model Selection and Inference."
  *Annals of Statistics* 33(1).

- Cribari-Neto (2004). "Asymptotic Inference Under
  Heteroskedasticity." *CSDA* 45(2).
                    """)),
                ], className="mb-4"),
            ], md=4),
        ]),
    ], fluid=True)
