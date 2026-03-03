"""
About / Documentation Page
===========================
Method explanation, algorithm details, DGP descriptions, references.
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/about", name="About",
                   title="AMRI \u2014 About")


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
weight that responds to the *SE ratio* \u2014 a natural diagnostic for misspecification.

### Algorithm (AMRI v2)

```
Input: X, Y, \u03b1 = 0.05, c\u2081 = 1.0, c\u2082 = 2.0

1. Fit OLS: Y = a + bX + \u03b5
2. Compute SE_naive (model-based) and SE_HC3 (sandwich robust)
3. R = SE_HC3 / SE_naive              (SE ratio)
4. s = |log(R)|                        (symmetric, scale-invariant diagnostic)
5. lo = c\u2081 / \u221an                        (lower threshold, shrinks with n)
6. hi = c\u2082 / \u221an                        (upper threshold)
7. w = clip((s \u2212 lo) / (hi \u2212 lo), 0, 1)  (blending weight)
8. SE_AMRI = (1\u2212w) \u00b7 SE_naive + w \u00b7 SE_HC3
9. CI = \u03b8\u0302 \u00b1 t_{n\u22122, 1\u2212\u03b1/2} \u00b7 SE_AMRI
```

### Key Properties

1. **Coverage Continuity** (Theorem 1): The map \u03b4 \u2192 Coverage is smooth
   (no discontinuous jumps, addressing the Leeb\u2013P\u00f6tscher pre-test critique)

2. **Asymptotic Validity** (Theorem 2): Coverage \u2265 1\u2212\u03b1 asymptotically
   for all \u03b4 \u2208 [0, \u03b4_max]

3. **Near-Oracle Efficiency** (Theorem 3): Under correct specification,
   width overhead is O(1/n) \u2014 negligible
                    """)),
                ], className="mb-4"),

                # DGP descriptions
                dbc.Card([
                    dbc.CardHeader(html.H5("Data Generating Processes (DGPs)",
                                           className="mb-0")),
                    dbc.CardBody(
                        dbc.Table([
                            html.Thead(html.Tr([
                                html.Th("DGP"), html.Th("Name"),
                                html.Th("Misspecification Type"),
                                html.Th("Formula"),
                            ])),
                            html.Tbody([
                                html.Tr([
                                    html.Td("1"), html.Td("Nonlinearity"),
                                    html.Td("Functional form"),
                                    html.Td("Y = X + \u03b4\u00b7X\u00b2 + \u03b5"),
                                ]),
                                html.Tr([
                                    html.Td("2"), html.Td("Heavy Tails"),
                                    html.Td("Distributional"),
                                    html.Td("\u03b5 ~ t(10 \u2212 8\u03b4)"),
                                ]),
                                html.Tr([
                                    html.Td("3"),
                                    html.Td("Heteroscedasticity"),
                                    html.Td("Variance misspecification"),
                                    html.Td("Var(\u03b5|X) = exp(\u03b4\u00b7X)"),
                                ]),
                                html.Tr([
                                    html.Td("4"),
                                    html.Td("Omitted Variable"),
                                    html.Td("Bias from omission"),
                                    html.Td("Y = X\u2081 + \u03b4\u00b7X\u2082 + \u03b5"),
                                ]),
                                html.Tr([
                                    html.Td("5"), html.Td("Clustering"),
                                    html.Td("Ignored dependence"),
                                    html.Td(
                                        "Y = X + \u221a\u03b4\u00b7u_g + \u03b5"),
                                ]),
                                html.Tr([
                                    html.Td("6"),
                                    html.Td("Contaminated"),
                                    html.Td("Outlier contamination"),
                                    html.Td(
                                        "\u03b5 ~ (1\u2212p)\u00b7N(0,1) "
                                        "+ p\u00b7N(0,25)"),
                                ]),
                            ]),
                        ], bordered=True, hover=True, responsive=True,
                           size="sm"),
                    ),
                ], className="mb-4"),
            ], md=8),

            dbc.Col([
                # Method descriptions
                dbc.Card([
                    dbc.CardHeader(html.H5("Inference Methods",
                                           className="mb-0")),
                    dbc.CardBody(dcc.Markdown("""
**Naive OLS** \u2014 Model-based SE assuming homoscedasticity.

**Sandwich HC3** \u2014 MacKinnon & White (1985). Stata's default robust SE.

**HC4** \u2014 Cribari-Neto (2004). Leverage-adjusted robust SE.

**HC5** \u2014 Cribari-Neto & da Silva (2011). Max-leverage-aware.

**Pairs Bootstrap** \u2014 Resample (X, Y) pairs, percentile CI.

**Wild Bootstrap** \u2014 Rademacher weights, preserves heteroscedasticity.

**Bootstrap-t** \u2014 Studentized bootstrap for refined coverage.

**AKS Adaptive** \u2014 Armstrong, Kline & Sun (2025, *Econometrica*).
Soft-thresholds toward efficient SE. Optimizes estimation regret.

**AMRI v1** \u2014 Hard-switching between naive and robust SE.

**AMRI v2** \u2014 Soft-threshold blending (recommended). Near-minimax
optimal for CI construction.
                    """)),
                ], className="mb-4"),

                # References
                dbc.Card([
                    dbc.CardHeader(html.H5("Key References",
                                           className="mb-0")),
                    dbc.CardBody(dcc.Markdown("""
- Armstrong, Kline & Sun (2025). \u201cAdapting to Misspecification.\u201d
  *Econometrica* 93(6): 1981\u20132005.

- White (1980). \u201cA Heteroskedasticity-Consistent Covariance Matrix
  Estimator.\u201d *Econometrica* 48(4).

- MacKinnon & White (1985). \u201cSome Heteroskedasticity-Consistent
  Covariance Matrix Estimators.\u201d *JBES* 3(3).

- Leeb & P\u00f6tscher (2005). \u201cModel Selection and Inference.\u201d
  *Annals of Statistics* 33(1).

- Cribari-Neto (2004). \u201cAsymptotic Inference Under
  Heteroskedasticity.\u201d *CSDA* 45(2).
                    """)),
                ], className="mb-4"),
            ], md=4),
        ]),
    ], fluid=True)
