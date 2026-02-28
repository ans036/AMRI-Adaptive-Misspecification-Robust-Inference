# AMRI: Adaptive Misspecification-Robust Inference

**A novel statistical method for confidence interval construction that automatically adapts to model misspecification.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Problem

Standard statistical inference faces a fundamental tradeoff:

| Approach | Model Correct | Model Wrong |
|----------|:------------:|:-----------:|
| **Naive OLS** | Optimal (narrowest CIs) | Catastrophic (coverage collapses to 35-77%) |
| **Robust methods** (sandwich SE, bootstrap) | Pays a width penalty | Protected |

Practitioners must choose *before seeing the data* — but rarely know which regime they're in.

## The Solution: AMRI

AMRI resolves this tradeoff using a **data-driven diagnostic** that detects misspecification in real time:

```
R = SE_HC3 / SE_naive
```

- **R ≈ 1.0** → Model is correct → Use efficient naive inference
- **R >> 1 or R << 1** → Model is wrong → Switch to robust inference

### Algorithm

```
AMRI(X, Y, alpha):
    1. Fit OLS → θ̂, residuals e
    2. Compute SE_naive and SE_HC3
    3. Diagnostic ratio: R = SE_HC3 / SE_naive
    4. Adaptive threshold: τ(n) = 1 + 2/√n
    5. IF R > τ(n) OR R < 1/τ(n):
           SE = SE_HC3 × 1.05        [robust mode]
       ELSE:
           SE = SE_naive              [efficient mode]
    6. CI = θ̂ ± t_{n-2, 1-α/2} × SE
```

## Key Results

### Simulation Study (240 scenarios, 8 methods, 2000 reps each)

| Metric | AMRI | Best Competitor | Advantage |
|--------|:----:|:--------------:|:---------:|
| Overall coverage | **0.949** | HC3: 0.945 | +0.4pp (p=0.003) |
| Coverage at δ=0 (correct model) | 0.948 | Naive: 0.947 | Matches |
| Coverage at δ=1 (severe misspec) | **0.945** | HC3: 0.936 | +0.9pp |
| Width overhead at δ=0 | 1.01× | — | Negligible |
| Degradation rate | **+0.007** | Boot-t: −0.003 | Only positive slope |

**AMRI is the only method whose coverage *improves* as misspecification gets worse.**

### Real Data Validation (11 datasets, 50K bootstrap ground truth)

| Dataset | n | Naive Coverage | AMRI Coverage | AMRI Mode |
|---------|--:|:-------------:|:------------:|:---------:|
| California Rooms | 20,640 | **0.355** | **0.965** | Robust |
| Star98 Math | 303 | 0.624 | 1.000 | Robust |
| Fair Affairs | 6,366 | 0.905 | 0.960 | Robust |
| Diabetes BMI | 442 | 0.966 | 0.966 | Efficient |
| Iris Sepal | 150 | 0.966 | 0.966 | Efficient |

**Average:** Naive 0.853, HC3 0.960, **AMRI 0.962**

### Adversarial Stress Testing

AMRI survived **10/10 deliberately hostile DGPs** (coverage ≥ 0.92 in all cases), including threshold-edge attacks, extreme heavy tails (t₂.₅), leverage outliers, and mixture-of-regressions.

## Statistical Guarantees

| Test | Result | p-value |
|------|--------|:-------:|
| TOST equivalence to 0.95 (ε=0.01) | **EQUIVALENT** | 0.00008 |
| AMRI > HC3 (paired t-test) | **SIGNIFICANT** | 0.0029 |
| AMRI > HC3 (permutation, 50K reps) | **SIGNIFICANT** | 0.0017 |
| Hochberg FWER correction (5 hypotheses) | **4/5 confirmed** | — |
| Cohen's d vs Naive OLS | **1.113** (large) | — |

**Formal asymptotic guarantee:** Under mild conditions (finite 4th moments), AMRI coverage → 1−α for *any* DGP.

## Project Structure

```
├── README.md                    # This file
├── HYPOTHESES.md                # Complete hypotheses, results, and proofs
├── AMRI_METHOD.md               # Standalone AMRI method documentation
├── RESEARCH_PLAN.md             # Initial research plan
├── EXPERIMENTAL_DESIGN.md       # Experimental design specification
├── src/
│   ├── simulation.py            # Core simulation engine (DGPs + methods)
│   ├── run_vectorized_v2.py     # Full vectorized Monte Carlo (1440 scenarios)
│   ├── run_full_vectorized.py   # Vectorized simulation (earlier version)
│   ├── run_optimized.py         # Optimized simulation runner
│   ├── run_fast.py              # Fast pilot simulation
│   ├── analyze_and_plot.py      # Analysis and visualization pipeline
│   ├── deep_analysis.py         # Deep 6-panel analysis figures
│   ├── amri_figure.py           # AMRI comprehensive comparison figure
│   ├── statistical_guarantees.py# 7 formal statistical tests
│   ├── generalization_proof.py  # 4-pillar generalization framework
│   ├── real_data_validation.py  # Real-world dataset validation (11 datasets)
│   └── reanalyze_complete.py    # One-click reanalysis script
├── results/
│   ├── results_pilot.csv        # Pilot results (60 scenarios)
│   └── results_intermediate.csv # Intermediate results (240 scenarios)
└── figures/
    ├── AMRI_comprehensive.png   # 6-panel AMRI comparison (main figure)
    ├── FINAL_1 through FINAL_6  # Publication-ready figures
    ├── A through F              # Deep analysis panels
    ├── fig1 through fig7        # Initial pilot figures
    └── *.csv                    # Summary statistics tables
```

## Reproducing Results

### Prerequisites

```bash
pip install numpy pandas scipy statsmodels scikit-learn matplotlib seaborn rdatasets
```

### Run Simulations

```bash
# Pilot (fast, ~5 min)
python src/run_fast.py

# Full simulation (1440 scenarios × 2000 reps, ~10 hours)
python -u src/run_vectorized_v2.py
```

### Generate Figures and Tests

```bash
# Deep analysis figures
python src/deep_analysis.py

# AMRI comprehensive figure
python src/amri_figure.py

# Formal statistical guarantees (7 tests)
python src/statistical_guarantees.py

# Generalization proof (theory + adversarial + hypothesis testing)
python src/generalization_proof.py

# Real-world data validation (11 datasets, 50K bootstrap)
python src/real_data_validation.py

# One-click reanalysis (after full simulation completes)
python -u src/reanalyze_complete.py
```

## Formal Hypotheses

| # | Hypothesis | Status | p-value |
|:-:|-----------|:------:|:-------:|
| H1 | Naive coverage degrades monotonically with misspecification | **Confirmed** | < 10⁻¹⁰ |
| H2 | Sandwich HC3 maintains coverage ≥ 0.93 universally | **Confirmed** | Simes NS |
| H3 | AMRI achieves best-of-both-worlds | **Confirmed** | 0.0007 |
| H4 | Degradation rate: Naive >> Bootstrap > Sandwich > AMRI | **Confirmed** | 0.00008 |
| H5 | Robust methods widen adaptively under misspecification | **Confirmed** | 0.011 |
| Bonus | More data hurts Naive but helps AMRI (sample size paradox) | **Confirmed** | 0.015 |

## Methods Compared

1. **Naive OLS** — Model-based standard errors
2. **Sandwich HC0** — Heteroscedasticity-consistent (White, 1980)
3. **Sandwich HC3** — Leverage-corrected (MacKinnon & White, 1985)
4. **Pairs Bootstrap** — Resample (Xᵢ, Yᵢ) pairs
5. **Wild Bootstrap** — Rademacher perturbation of residuals
6. **Bootstrap-t** — Studentized bootstrap
7. **Bayesian** — Normal-Gamma posterior with vague priors
8. **AMRI** — Adaptive Misspecification-Robust Inference (proposed)

## Data Generating Processes

| DGP | Misspecification Type | How δ Controls Severity |
|:---:|----------------------|------------------------|
| 1 | Nonlinearity | Y = X + δ·X² + ε |
| 2 | Heavy-tailed errors | ε ~ (1-δ)·N(0,1) + δ·t(3) |
| 3 | Heteroscedasticity | Var(ε\|X) = (1 + δ·X²) |
| 4 | Omitted variable | Y = X + δ·Z + ε, Z⊥X |
| 5 | Clustering | ICC increases with δ |
| 6 | Contaminated normal | ε ~ (1-δ)·N(0,1) + δ·N(0,25) |

## Citation

If you use this code or method, please cite:

```bibtex
@software{amri2026,
  title={AMRI: Adaptive Misspecification-Robust Inference},
  author={Anish},
  year={2026},
  url={https://github.com/ans036/AMRI-Adaptive-Misspecification-Robust-Inference}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
