# Experimental Design Specification
## Inference Under Model Misspecification — Simulation & Empirical Study

---

## 1. SIMULATION STUDY ARCHITECTURE (ADEMP Framework)

Following Morris, White & Crowther (2019) "Using simulation studies to evaluate statistical methods":

### A — AIMS

| Aim | Description |
|-----|------------|
| **A1** | Quantify how coverage probability degrades under each type of misspecification for each inference method |
| **A2** | Characterize the efficiency-robustness tradeoff (interval width vs. coverage) |
| **A3** | Identify which method is best under each misspecification type and severity |
| **A4** | Demonstrate that the proposed AMRI method achieves near-optimal behavior across all scenarios |
| **A5** | Provide clear practitioner recommendations |

### D — DATA-GENERATING MECHANISMS

#### DGP1: Functional Form Misspecification (Nonlinearity)

```
True model: Y = β₀ + β₁X + δ·β₂X² + ε,  ε ~ N(0, 1)
Fitted model: Y = α₀ + α₁X + error  (linear)
Parameters: β₀ = 0, β₁ = 1, β₂ = 1, X ~ N(0, 1)
Severity: δ ∈ {0, 0.25, 0.5, 0.75, 1.0}
  - δ=0: correctly specified (Y = X + ε)
  - δ=1: Y = X + X² + ε (substantial nonlinearity)
Pseudo-true parameter: α₁* = β₁ + δ·β₂·E[X²·X]/E[X²]
  = 1 + δ·E[X³]/E[X²] = 1 (by symmetry of X~N(0,1))
  Actually: α₁* = Cov(Y,X)/Var(X) = β₁ + δ·β₂·E[X³] = 1 (since E[X³]=0)
  But the model is still wrong because E[Y|X] ≠ α₀* + α₁*X
```

#### DGP2: Distributional Misspecification (Heavy Tails)

```
True model: Y = β₀ + β₁X + ε
Fitted model: Same mean structure, assumes ε ~ N(0, σ²)
Error distributions (indexed by δ):
  - δ=0: ε ~ N(0, 1) (correct)
  - δ=0.25: ε ~ t(df=10) (mild heavy tails)
  - δ=0.5: ε ~ t(df=5) (moderate)
  - δ=0.75: ε ~ t(df=3) (heavy)
  - δ=1.0: ε ~ t(df=2) (very heavy, infinite variance)
Parameters: β₀ = 0, β₁ = 1, X ~ N(0, 1)
Note: Point estimate (OLS) is still consistent, but
      model-based SE is wrong.
```

#### DGP3: Heteroscedasticity

```
True model: Y = β₀ + β₁X + ε,  ε ~ N(0, σ²(X))
             where σ²(X) = exp(δ · γ · X)
Fitted model: Y = α₀ + α₁X + error, assumes homoscedasticity
Parameters: β₀ = 0, β₁ = 1, γ = 1, X ~ N(0, 1)
Severity: δ ∈ {0, 0.25, 0.5, 0.75, 1.0}
  - δ=0: homoscedastic (correct)
  - δ=1: Var(ε|X) = exp(X), strong heteroscedasticity
```

#### DGP4: Omitted Variable Bias

```
True model: Y = β₁X₁ + β₂X₂ + ε,  ε ~ N(0, 1)
Fitted model: Y = α₁X₁ + error  (omits X₂)
Joint distribution: (X₁, X₂) ~ N(0, Σ), Σ₁₂ = ρ
Parameters: β₁ = 1, β₂ = 1, ρ = 0.5
Severity: δ controls β₂: β₂ = δ  (δ ∈ {0, 0.25, 0.5, 0.75, 1.0})
  - δ=0: X₂ irrelevant (correct model)
  - δ=1: substantial omitted variable effect
Pseudo-true parameter: α₁* = β₁ + δ·β₂·ρ = 1 + δ·0.5
  (the coefficient is BIASED; inference about β₁ is misleading)
Target of inference: β₁ = 1 (the causal parameter of interest)
```

#### DGP5: Link Function Misspecification

```
True model: P(Y=1|X) = Φ(β₀ + β₁X)  (Probit)
Fitted model: P(Y=1|X) = logit⁻¹(α₀ + α₁X)  (Logit)
Parameters: β₀ = 0, X ~ N(0, 1)
Severity: β₁ = 0.5 + 2δ  (larger β₁ = more divergence between probit/logit)
  - δ=0: β₁=0.5 (near-identical probit/logit)
  - δ=1: β₁=2.5 (substantial divergence in tails)
Note: Probit and logit are nearly indistinguishable for small |Xβ|
      but diverge in the tails.
```

#### DGP6: Ignored Clustering / Dependence

```
True model: Y_ij = β₀ + β₁X_ij + u_i + ε_ij
            u_i ~ N(0, τ²), ε_ij ~ N(0, σ²)
            i = 1,...,G clusters, j = 1,...,m within cluster
Fitted model: Y_ij = α₀ + α₁X_ij + error  (ignores clustering)
Parameters: β₀ = 0, β₁ = 1, σ² = 1
Severity: τ² = δ  (δ ∈ {0, 0.25, 0.5, 0.75, 1.0})
  - δ=0: no clustering (correct)
  - δ=1: ICC = τ²/(τ²+σ²) = 0.5 (strong clustering)
G = 50 clusters, m = 10 per cluster (n = 500)
For varying n: adjust G proportionally, keep m = 10
```

### E — ESTIMANDS

| DGP | Primary Estimand | Definition |
|-----|-----------------|------------|
| DGP1 | Slope coefficient | α₁* = Cov(Y,X)/Var(X) (projection parameter) |
| DGP2 | Slope coefficient | β₁ = 1 (correctly identified) |
| DGP3 | Slope coefficient | β₁ = 1 (correctly identified, but SE wrong) |
| DGP4 | Slope coefficient | Target: β₁ = 1; Pseudo-true: α₁* = 1 + 0.5δ |
| DGP5 | Log-odds ratio | Pseudo-true logit coefficient approximating probit |
| DGP6 | Slope coefficient | β₁ = 1 (correctly identified, but SE wrong) |

### M — METHODS

#### Method 1: Naive MLE + Model-Based SE
```python
from statsmodels.api import OLS
model = OLS(Y, X).fit()  # Uses model-based SE by default
ci = model.conf_int(alpha=0.05)
```

#### Method 2: Sandwich SE (HC3)
```python
model = OLS(Y, X).fit(cov_type='HC3')
ci = model.conf_int(alpha=0.05)
```

#### Method 3: Pairs Bootstrap (Percentile)
```python
# Resample (X_i, Y_i) pairs B_boot times
# Compute percentile CI from bootstrap distribution
B_boot = 999
theta_boot = [OLS(Y[idx], X[idx]).fit().params[1]
              for idx in bootstrap_indices(n, B_boot)]
ci = np.percentile(theta_boot, [2.5, 97.5])
```

#### Method 4: Wild Bootstrap
```python
# Fit model, get residuals e_i
# For each bootstrap rep: Y*_i = X_i @ beta_hat + e_i * v_i
# where v_i ~ Rademacher({-1, +1} with prob 0.5 each)
```

#### Method 5: Bootstrap-t (Studentized)
```python
# For each bootstrap rep, compute t* = (theta*_boot - theta_hat) / SE*_boot
# CI: [theta_hat - q_{1-α/2} * SE_hat, theta_hat - q_{α/2} * SE_hat]
```

#### Method 6: Split Conformal Prediction
```python
from mapie.regression import MapieRegressor
mapie = MapieRegressor(estimator=LinearRegression(), method='base',
                        cv='split')
mapie.fit(X_train, Y_train)
y_pred, y_pis = mapie.predict(X_test, alpha=0.05)
```

#### Method 7: Conformalized Quantile Regression (CQR)
```python
from mapie.regression import MapieQuantileRegressor
mapie_cqr = MapieQuantileRegressor(estimator=QuantileRegressor(),
                                     cv='split')
```

#### Method 8: Jackknife+
```python
mapie_jp = MapieRegressor(estimator=LinearRegression(),
                           method='plus', cv=-1)  # LOO
```

#### Method 9: Standard Bayesian Posterior
```python
import pymc as pm
with pm.Model():
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)
    sigma = pm.HalfCauchy('sigma', beta=5)
    mu = alpha + beta * X
    Y_obs = pm.Normal('Y', mu=mu, sigma=sigma, observed=Y)
    trace = pm.sample(2000, tune=1000)
# CI = pm.hdi(trace, hdi_prob=0.95)['beta']
```

#### Method 10: Generalized Bayes (Gibbs Posterior)
```python
# Tempered likelihood: L(θ|data)^η, η < 1
# η chosen by cross-validation or SafeBayes criterion
# Implemented via modified PyMC or custom MCMC
```

#### Method 11: Coarsened Posterior (Miller & Dunson 2019)
```python
# Replace P(data|θ) with P(data ∈ B_ε(generated data))
# where B_ε is an ε-ball in some metric
# Implementation: importance sampling with soft likelihood
```

#### Method 12: Universal Inference (Split LRT)
```python
# Split data into D1 (estimation) and D2 (inference)
# Compute: e(θ₀) = L(θ̂_D1 | D2) / L(θ₀ | D2)
# Confidence set: {θ₀ : e(θ₀) ≤ 1/α}
# Valid without ANY regularity conditions
```

#### Method 13: AMRI (Proposed Hybrid Method)
```python
def AMRI(X, Y, alpha=0.05):
    # Step 1: Fit model, get naive and sandwich CIs
    model = OLS(Y, X).fit()
    ci_naive = model.conf_int(alpha=alpha)
    ci_sandwich = OLS(Y, X).fit(cov_type='HC3').conf_int(alpha=alpha)

    # Step 2: Misspecification diagnostic
    # Compare model-based SE vs sandwich SE (ratio test)
    se_ratio = sandwich_se / model_se
    misspec_detected = abs(se_ratio - 1) > threshold(n)

    # Step 3: Adaptive inference
    if misspec_detected:
        # Use sandwich CI, but also check conformal
        ci_conformal = conformal_CI(X, Y, alpha)
        # Return union for safety
        ci = [min(ci_sandwich[0], ci_conformal[0]),
              max(ci_sandwich[1], ci_conformal[1])]
    else:
        ci = ci_naive  # More efficient when model seems correct

    return ci
```

### P — PERFORMANCE MEASURES

```python
def compute_metrics(results, theta_true):
    """
    results: array of shape (B, 4) = [theta_hat, se_hat, ci_lower, ci_upper]
    theta_true: the pseudo-true parameter value
    """
    # Coverage
    coverage = np.mean((results[:, 2] <= theta_true) &
                        (theta_true <= results[:, 3]))

    # Monte Carlo SE of coverage
    mc_se_coverage = np.sqrt(coverage * (1 - coverage) / len(results))

    # Average width
    avg_width = np.mean(results[:, 3] - results[:, 2])

    # Bias
    bias = np.mean(results[:, 0] - theta_true)

    # RMSE
    rmse = np.sqrt(np.mean((results[:, 0] - theta_true) ** 2))

    # Relative efficiency (vs oracle)
    # rel_eff = mse_method / mse_oracle

    return {
        'coverage': coverage,
        'mc_se_coverage': mc_se_coverage,
        'avg_width': avg_width,
        'bias': bias,
        'rmse': rmse
    }
```

---

## 2. FULL FACTORIAL DESIGN

### Design Matrix

| Factor | Levels | Count |
|--------|--------|-------|
| DGP | DGP1–DGP6 | 6 |
| Severity (δ) | 0, 0.25, 0.5, 0.75, 1.0 | 5 |
| Sample size (n) | 50, 100, 250, 500, 1000, 5000 | 6 |
| Method | Methods 1–13 | 13 |
| Replications (B) | — | 5,000 |

**Total scenarios**: 6 × 5 × 6 × 13 = **2,340 scenarios**
**Total simulation runs**: 2,340 × 5,000 = **11,700,000 runs**

### Computational Budget

Estimated per-run time:
- Methods 1-5 (frequentist): ~0.01-0.5 sec each
- Methods 6-8 (conformal): ~0.1-1 sec each
- Methods 9-11 (Bayesian): ~5-30 sec each (MCMC)
- Methods 12-13 (universal/AMRI): ~0.5-2 sec each

**Estimated total compute**:
- Fast methods (1-8, 12-13): ~11.7M × 0.5s / 10 methods × average ≈ 1,600 CPU-hours
- Bayesian methods (9-11): ~11.7M × 15s / 3 methods × average ≈ 16,000 CPU-hours
- **Total: ~18,000 CPU-hours ≈ 750 CPU-days**

With 100-core parallel cluster: **~7.5 days**
With Ray on cloud (500 cores): **~1.5 days**

### Parallelization Strategy

```python
import ray

@ray.remote
def run_single_scenario(dgp, severity, n, method, seed):
    """Run one scenario for all B replications."""
    np.random.seed(seed)
    results = []
    for b in range(B):
        data = generate_data(dgp, severity, n)
        result = run_method(method, data)
        results.append(result)
    return compute_metrics(np.array(results), get_pseudo_true(dgp, severity))

# Launch all scenarios in parallel
futures = []
for dgp in DGPs:
    for delta in severities:
        for n in sample_sizes:
            for method in methods:
                futures.append(
                    run_single_scenario.remote(dgp, delta, n, method, seed=...)
                )

results = ray.get(futures)  # 2,340 result objects
```

---

## 3. VISUALIZATION PLAN

### Figure 1: Coverage vs. Sample Size (Main Result)

```
Layout: 6 × 5 facet grid (DGP × severity)
X-axis: Sample size (log scale)
Y-axis: Coverage probability
Lines: One per method (colored)
Horizontal reference: 0.95 (dashed red)
Gray band: [0.94, 0.96] (acceptable coverage zone)
```

### Figure 2: Coverage vs. Misspecification Severity

```
Layout: 6 panels (one per DGP)
X-axis: Severity δ (0 to 1)
Y-axis: Coverage probability
Lines: One per method
Fixed n = 500
Key finding: Where does each method's coverage break down?
```

### Figure 3: Efficiency-Robustness Frontier

```
Layout: Scatter plot
X-axis: Average interval width (efficiency)
Y-axis: Coverage probability (robustness)
Points: One per method, colored and shaped
Facets: By DGP
The "Pareto frontier" identifies methods with best tradeoff
```

### Figure 4: QQ-Plots of Z-Scores

```
Layout: 13 × 6 grid (method × DGP)
Plot: QQ-plot of (θ̂ - θ*)/SE vs N(0,1)
Departure from diagonal = inferential invalidity
```

### Figure 5: AMRI vs. Best Alternatives

```
Layout: Paired comparison plots
Show coverage and width of AMRI vs. each alternative
Highlight regions where AMRI dominates
```

### Table 1: Summary of All Results

```
Rows: Methods (13)
Column groups: DGP1 through DGP6
Within each group: Coverage | Width (at n=500, δ=0.5)
Bold: Coverage within [0.94, 0.96]
Italic: Best width among methods with valid coverage
```

---

## 4. REAL DATA ANALYSIS DESIGN

### Application 1: Lalonde Jobs Training Data

**Dataset**: NSW (National Supported Work) experimental + CPS/PSID observational controls
- Treatment: Job training program
- Outcome: Real earnings in 1978
- Ground truth: Experimental estimate ≈ $1,794

**Analysis**:
1. Fit various outcome models (linear, quadratic, GAM)
2. For each model, compute treatment effect + CI using all 13 methods
3. Compare CIs: which methods' CIs contain the experimental estimate?
4. Key question: Does sandwich/conformal/AMRI provide valid inference where naive fails?

**Misspecification present**: The linear model for earnings is certainly wrong (nonlinearity, selection bias, heteroscedasticity).

### Application 2: Medical Expenditure Panel Survey (MEPS)

**Dataset**: MEPS 2019, ~30,000 individuals
- Outcome: Total medical expenditure (heavily right-skewed)
- Predictors: Age, gender, insurance, chronic conditions, etc.

**Analysis**:
1. Fit linear regression (clearly misspecified for skewed outcome)
2. Fit log-linear model (better but still wrong)
3. Compare prediction intervals from all methods
4. Key metric: Coverage of held-out test data

**Misspecification present**: Any parametric model for medical expenditure is wrong (zero-inflation, extreme skewness, heteroscedasticity).

### Application 3: Riboflavin Genomics (High-Dimensional)

**Dataset**: Riboflavin (vitamin B2) production data
- n = 71 observations, p = 4,088 gene expression predictors
- Outcome: Riboflavin production rate

**Analysis**:
1. Fit Lasso/Ridge (sparsity/shrinkage assumptions may be wrong)
2. Apply debiased Lasso + sandwich SE
3. Compare with conformal prediction intervals
4. Test: Which genes have valid CIs for their effects?

**Misspecification present**: Linear model with sparsity is certainly wrong for gene regulation.

---

## 5. REPRODUCIBILITY SPECIFICATION

### Random Seed Strategy
```python
MASTER_SEED = 20260228  # Date-based for reproducibility
rng = np.random.SeedSequence(MASTER_SEED)
child_seeds = rng.spawn(n_scenarios)  # One per scenario
```

### Code Structure
```
inference-misspecification/
├── README.md
├── requirements.txt / environment.yml
├── Makefile
├── src/
│   ├── dgp/                    # Data generating processes
│   │   ├── dgp1_nonlinearity.py
│   │   ├── dgp2_heavy_tails.py
│   │   ├── dgp3_heteroscedasticity.py
│   │   ├── dgp4_omitted_variable.py
│   │   ├── dgp5_link_function.py
│   │   └── dgp6_clustering.py
│   ├── methods/                # Inference methods
│   │   ├── naive_mle.py
│   │   ├── sandwich.py
│   │   ├── bootstrap.py
│   │   ├── conformal.py
│   │   ├── bayesian.py
│   │   ├── universal_inference.py
│   │   └── amri.py
│   ├── metrics/                # Performance evaluation
│   │   └── compute_metrics.py
│   ├── simulation/             # Simulation runner
│   │   └── runner.py
│   └── visualization/          # Plotting code
│       ├── coverage_plots.py
│       ├── efficiency_plots.py
│       └── qq_plots.py
├── data/                       # Real datasets
│   ├── lalonde/
│   ├── meps/
│   └── riboflavin/
├── results/                    # Simulation output
│   ├── raw/
│   └── processed/
├── figures/                    # Generated figures
└── paper/                      # LaTeX manuscript
    ├── main.tex
    ├── supplement.tex
    └── references.bib
```

### Makefile
```makefile
.PHONY: all simulate analyze figures paper clean

all: simulate analyze figures paper

simulate:
    python src/simulation/runner.py --config configs/full_simulation.yaml

analyze:
    python src/simulation/analyze_results.py --input results/raw/ --output results/processed/

figures:
    python src/visualization/make_all_figures.py

paper:
    cd paper && latexmk -pdf main.tex

clean:
    rm -rf results/raw/* results/processed/* figures/*
```

---

## 6. PILOT STUDY PLAN

Before the full simulation, run a pilot:

### Pilot Specification
- **DGPs**: DGP1 (nonlinearity) and DGP3 (heteroscedasticity) only
- **Severity**: δ ∈ {0, 0.5, 1.0} only
- **Sample sizes**: n ∈ {100, 500} only
- **Methods**: Naive, Sandwich, Bootstrap-t, Conformal, Standard Bayes (5 methods)
- **Replications**: B = 1,000
- **Total runs**: 2 × 3 × 2 × 5 × 1,000 = 60,000

### Pilot Goals
1. Verify code correctness (δ=0 should give nominal coverage)
2. Estimate run times to plan full simulation
3. Identify any numerical issues (e.g., Bayesian convergence)
4. Preview key results to guide paper narrative

### Pilot Timeline: 1-2 days

---

*This document provides the complete specification needed to implement the simulation study. All DGPs, methods, metrics, and visualization are fully specified for reproducible research.*
