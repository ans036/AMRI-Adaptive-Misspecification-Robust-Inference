# Research Plan: Inference Under Model Misspecification
## "What Can We Still Say When the Model Is Only an Approximation or Possibly Wrong?"

---

## 1. EXECUTIVE SUMMARY

This research addresses one of the most fundamental questions in statistics and machine learning: **when the statistical model we use is wrong (as all models inevitably are), what can we still validly conclude?**

The proposal has two complementary framings:
1. **Approximate/wrong model with inference under weak or violated assumptions**
2. **Inference under weak or violated assumptions asks a different question → what can we still say?**

We propose a comprehensive study that unifies frequentist, Bayesian, and distribution-free approaches to inference under misspecification, with novel contributions at identified gaps in the literature.

---

## 2. BACKGROUND & MOTIVATION

### 2.1 The Core Problem

George Box famously said: *"All models are wrong, but some are useful."* Yet standard statistical inference—confidence intervals, p-values, posterior credible intervals—is derived under the assumption that the model is **exactly correct**. When this assumption is violated:

- **95% confidence intervals may cover the true parameter only 50-70% of the time**
- **Bayesian posteriors concentrate at wrong values with wrong uncertainty**
- **P-values become meaningless** (neither conservative nor anti-conservative in predictable ways)
- **Causal effect estimates can be severely biased**

### 2.2 Why This Matters Now More Than Ever

Three converging forces make this research timely:

1. **Foundation Models & AI Deployment**: Every deployed ML model faces distribution shift (misspecification). LLMs used in high-stakes domains (medical, legal) are misspecified relative to deployment context.

2. **Distribution-Free Methods Have Matured**: Conformal prediction, universal inference, e-values now provide practical alternatives to model-dependent inference.

3. **AI Safety Connection**: Recent work (Xu et al. 2026, "Epistemic Traps") shows model misspecification in world models leads to rational-but-unsafe behavior—connecting misspecification theory to alignment.

### 2.3 The Philosophical Reframing

The key insight from the proposal is the **change of question**:

| Traditional Question | New Question |
|---------------------|-------------|
| "Is my model correct?" | "What can I still say given my model is wrong?" |
| "How do I fix the model?" | "How do I fix the inference?" |
| "What is the true parameter?" | "What is the best approximation parameter, and how uncertain am I about it?" |

This reframing—from **model repair** to **inference repair**—is the central philosophical contribution.

---

## 3. LITERATURE LANDSCAPE

### 3.1 Foundational Works (The Classical Pillars)

| Paper | Key Contribution |
|-------|-----------------|
| **Huber (1967)** — "The behavior of ML estimates under nonstandard conditions" | MLEs consistent under misspecification; sandwich variance formula |
| **White (1980)** — "A heteroscedasticity-consistent covariance matrix estimator" | Practical sandwich (HC) standard errors |
| **White (1982)** — "Maximum likelihood estimation of misspecified models" | Pseudo-true parameter (KL minimizer); Information Matrix test |
| **Liang & Zeger (1986)** — "Longitudinal data analysis using GLMs" | GEE: valid inference with wrong working correlation |
| **Berk (1966)** — "Limiting behavior of posterior distributions when model is incorrect" | Bayesian posterior concentrates on KL minimizer |
| **Kleijn & van der Vaart (2012)** — "BvM theorem under misspecification" | Posterior variance is WRONG under misspecification |
| **Buja et al. (2019a,b)** — "Models as Approximations I & II" | Modern framework: parameters as population projections |

### 3.2 Modern Frameworks

#### A. Frequentist Robust Inference
- Sandwich estimators (HC0–HC5, cluster-robust, HAC)
- Bootstrap methods (pairs, wild, cluster)
- Universal inference (Wasserman, Ramdas, Balakrishnan 2020)

#### B. Distribution-Free Methods
- Conformal prediction (Vovk et al. 2005; Barber et al. 2021, 2023)
- Conformalized quantile regression (Romano, Patterson, Candes 2019)
- Jackknife+ (Barber et al. 2021)

#### C. Bayesian Approaches Under Misspecification
- Generalized/Gibbs posteriors (Bissiri, Holmes, Walker 2016)
- SafeBayes / tempered posteriors (Grunwald & van Ommen 2017)
- Coarsened posteriors (Miller & Dunson 2019)
- Martingale posteriors (Fong & Holmes 2023)

#### D. Game-Theoretic / E-Value Approaches
- E-values and safe testing (Grunwald, de Heide, Koolen 2024)
- Game-theoretic statistics (Ramdas et al. 2023)

#### E. Causal Inference Under Misspecification
- Double/debiased ML (Chernozhukov et al. 2018)
- Doubly robust estimation (Robins & Rotnitzky 1995)
- Sensitivity analysis (Rosenbaum 2002; Cinelli & Hazlett 2020)

### 3.3 Key Researchers

| Researcher | Affiliation | Focus |
|-----------|------------|-------|
| Larry Wasserman | CMU | Universal inference, distribution-free methods |
| Aaditya Ramdas | CMU | E-values, anytime-valid inference, conformal |
| Peter Grunwald | CWI Amsterdam/Leiden | SafeBayes, MDL, safe testing |
| Rina Foygel Barber | U. Chicago | Conformal prediction, distribution-free inference |
| Emmanuel Candes | Stanford | Conformal prediction, knockoffs |
| Peter Buhlmann | ETH Zurich | Invariance, causality, robustness |
| Victor Chernozhukov | MIT | Double/debiased ML |
| Mark van der Laan | UC Berkeley | TMLE, targeted learning |

---

## 4. IDENTIFIED GAPS & NOVEL CONTRIBUTION OPPORTUNITIES

### 4.1 Tier 1: Breakthrough Potential

**(A) E-Values Under Model Misspecification** ⭐ RECOMMENDED
- E-values have been developed under well-specified models
- **No systematic work** on what happens to e-value guarantees when the null is specified through a misspecified model
- Unifying two of the hottest recent developments (e-values + misspecification)
- Target: JRSS-B or Annals of Statistics

**(B) Fairness Guarantees Under Model Misspecification**
- Almost completely empty intersection (~5 papers total)
- What happens to demographic parity, equalized odds, counterfactual fairness when the model is wrong?
- Enormous practical need (fairness audits always use wrong models)
- Target: FAccT or NeurIPS

**(C) Unified Framework: "The Misspecification Spectrum"**
- No paper systematically compares ALL major approaches (sandwich, bootstrap, conformal, generalized Bayes, e-values, universal inference) on the SAME problems
- A comprehensive empirical + theoretical comparison paper would be highly cited
- Target: Statistical Science (survey) or JASA

### 4.2 Tier 2: Strong Contributions

**(D) Doubly Robust Variance Estimation Fix**
- Wu et al. (2025) showed DR variance estimation is NOT doubly robust
- A fix would immediately impact thousands of applied papers

**(E) Heterogeneous Misspecification in Federated Learning**
- Only ~3 papers exist; wide open territory
- Different clients misspecified in different ways

**(F) Universal Inference Meets Conformal Prediction**
- Both provide distribution-free guarantees from different angles
- Merging could yield more powerful procedures

---

## 5. PROPOSED RESEARCH: UNIFIED STUDY

### 5.1 Title Options

1. *"What Can We Still Say? A Unified Framework for Inference Under Model Misspecification"*
2. *"Inference Despite Misspecification: Comparing Frequentist, Bayesian, and Distribution-Free Approaches"*
3. *"The Misspecification Spectrum: From Sandwich Estimators to Conformal Prediction"*
4. *"Beyond Correct Models: A Comprehensive Study of Robust Inference Methods"*

### 5.2 Core Research Questions

**RQ1**: How do different inference methods (sandwich SE, bootstrap, conformal, generalized Bayes, e-values, universal inference) compare under varying types and severities of misspecification?

**RQ2**: Can we provide a unified theoretical framework that explains WHY some methods are robust to misspecification while others fail?

**RQ3**: What are practical guidelines for practitioners choosing among these methods?

**RQ4**: Are there novel hybrid methods that combine the strengths of multiple approaches?

### 5.3 Theoretical Contributions

1. **A taxonomy of misspecification types** with formal definitions and relationships
2. **Unified conditions** under which each method provides valid inference
3. **Impossibility results**: when NO method can provide valid inference (lower bounds)
4. **Novel hybrid method**: combining model-based efficiency with distribution-free safety nets

### 5.4 Methodological Innovation

**Proposed new method: "Adaptive Misspecification-Robust Inference" (AMRI)**

The key idea: use a model for efficiency, but automatically detect and correct for misspecification using distribution-free calibration.

```
Algorithm AMRI:
1. Fit parametric model → get point estimate θ̂ and model-based CI
2. Compute misspecification diagnostic (IM test, calibration check)
3. If misspecification detected:
   a. Compute sandwich-corrected CI
   b. Compute conformal prediction interval
   c. Return the intersection/union (conservative/liberal choice)
4. If no misspecification detected:
   a. Return model-based CI (more efficient)
   b. But provide conformal guarantee as backup
```

This bridges the "efficiency when correct, validity when wrong" gap (Open Problem 3 from the literature).

---

## 6. EXPERIMENTAL DESIGN

### 6.1 Simulation Study Design

#### Data Generating Processes (DGPs)

We use 6 canonical DGPs, each isolating a specific misspecification type:

| DGP | True Model | Fitted Model | Misspecification Type |
|-----|-----------|-------------|----------------------|
| **DGP1**: Nonlinearity | Y = X + 0.5X² + ε | Y ~ X (linear) | Functional form |
| **DGP2**: Heavy tails | ε ~ t(df=3) | Assume ε ~ N(0,σ²) | Distributional |
| **DGP3**: Heteroscedasticity | Var(ε\|X) = exp(γX) | Assume homoscedastic | Variance structure |
| **DGP4**: Omitted variable | Y = X₁β₁ + X₂β₂ + ε, cor(X₁,X₂)=ρ | Y ~ X₁ only | Omitted variable |
| **DGP5**: Wrong link | Y ~ Probit(Xβ) | Fit Logit(Xβ) | Link function |
| **DGP6**: Ignored clustering | Y_ij = X_ij β + u_i + ε_ij | Fit ignoring u_i | Dependence |

#### Misspecification Severity

Each DGP has a severity parameter δ ∈ {0, 0.25, 0.5, 0.75, 1.0} where:
- δ = 0: model is correctly specified (calibration baseline)
- δ = 1: severe misspecification

Example: For DGP1, Y = X + δ·X² + ε (δ controls nonlinearity strength)

#### Sample Sizes
n ∈ {50, 100, 250, 500, 1000, 5000}

#### Number of Replications
B = 5,000 per scenario (MC SE ≈ 0.003 for coverage)

### 6.2 Methods Compared

| # | Method | Category | Key Reference |
|---|--------|----------|--------------|
| 1 | **Naive MLE + model-based SE** | Baseline (breaks) | — |
| 2 | **Sandwich SE (HC3)** | Frequentist robust | White (1980), MacKinnon & White (1985) |
| 3 | **Pairs bootstrap (percentile)** | Bootstrap | Efron & Tibshirani (1993) |
| 4 | **Wild bootstrap** | Bootstrap | Wu (1986) |
| 5 | **Bootstrap-t (studentized)** | Bootstrap | Davison & Hinkley (1997) |
| 6 | **Split conformal** | Distribution-free | Lei et al. (2018) |
| 7 | **Conformalized quantile regression** | Distribution-free | Romano et al. (2019) |
| 8 | **Jackknife+** | Distribution-free | Barber et al. (2021) |
| 9 | **Standard Bayesian posterior** | Bayesian (breaks) | — |
| 10 | **Generalized Bayes (Gibbs posterior)** | Bayesian robust | Bissiri et al. (2016) |
| 11 | **Coarsened posterior** | Bayesian robust | Miller & Dunson (2019) |
| 12 | **Universal inference (split LRT)** | Nonparametric | Wasserman et al. (2020) |
| 13 | **AMRI (our proposed method)** | Hybrid (novel) | This paper |

### 6.3 Evaluation Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Coverage probability** | P(θ* ∈ CI) | 0.95 |
| **Average interval width** | E[upper - lower] | Smaller is better |
| **Bias** | E[θ̂ - θ*] | 0 |
| **RMSE** | √E[(θ̂ - θ*)²] | Smaller is better |
| **Relative efficiency** | MSE(method)/MSE(oracle) | Close to 1 |
| **QQ-plot of z-scores** | (θ̂ - θ*)/SE vs N(0,1) | Straight line |

**Critical note**: θ* is the **pseudo-true parameter** (KL minimizer), not the parameter of the true DGP (which may not exist in the model space).

### 6.4 Real Data Applications

#### Application 1: Lalonde Jobs Training Data (Causal Inference)
- Classic benchmark where different model specs give wildly different treatment effects
- Compare all methods for estimating the Average Treatment Effect
- Ground truth available from the experimental arm

#### Application 2: Medical Expenditure Panel Survey (MEPS)
- Heavily skewed, heteroscedastic outcome
- Used by Romano et al. (2019) for conformal prediction
- Compare prediction intervals under misspecified models

#### Application 3: Riboflavin Genomics Data (High-Dimensional)
- p=4088 predictors, n=71 observations
- Linear model certainly misspecified
- Tests methods in the p >> n regime

### 6.5 Software Implementation

**Primary language**: Python + R hybrid

| Component | Language | Packages |
|-----------|----------|----------|
| Simulation infrastructure | Python | `numpy`, `scipy`, `joblib`, `ray` |
| Sandwich estimators | R | `sandwich`, `lmtest`, `clubSandwich` |
| Bootstrap | R + Python | `boot`, `fwildclusterboot`, `arch` |
| Conformal prediction | Python | `mapie`, `crepes` |
| Bayesian methods | Python | `PyMC`, `numpyro` |
| Universal inference | R | Custom implementation |
| Visualization | Python + R | `matplotlib`, `seaborn`, `ggplot2` |
| Simulation framework | R | `SimDesign`, `rsimsum` |

---

## 7. PAPER STRUCTURE (Full-Length Research Paper)

### Target Venue: JASA (Theory & Methods) or Statistical Science

### Estimated Length: 30-40 pages (main) + 20-30 pages (supplement)

```
MAIN PAPER (~35 pages)

1. INTRODUCTION (3-4 pages)
   1.1 Motivating Example: "The 95% CI that covers only 62% of the time"
       - Concrete numerical example showing CI failure under misspecification
   1.2 The Core Question: What can we still say?
   1.3 Our Contributions (4 bullet points):
       • Unified taxonomy of misspecification types
       • Comprehensive comparison of 13 inference methods
       • Theoretical conditions for validity under each misspecification type
       • Novel AMRI hybrid method
   1.4 Organization

2. FRAMEWORK AND DEFINITIONS (3-4 pages)
   2.1 Formal Setup
       - Data X₁,...,Xn ~ P₀ (true, unknown)
       - Model family {Pθ : θ ∈ Θ}
       - P₀ ∉ {Pθ} (misspecification)
   2.2 The Pseudo-True Parameter
       - θ* = argmin KL(P₀ || Pθ)
       - What θ* means: "best approximation" in the model
       - Dependence on the divergence choice
   2.3 Taxonomy of Misspecification
       - Type I: Distributional (wrong error distribution)
       - Type II: Structural (wrong functional form)
       - Type III: Omission (missing variables)
       - Type IV: Dependence (wrong correlation structure)
       - Type V: Compositional (multiple simultaneous)
   2.4 What "Valid Inference" Means Under Misspecification
       - Frequentist: coverage of θ* at nominal level
       - Bayesian: calibrated posterior for θ*
       - Predictive: coverage of Y_{n+1}

3. EXISTING APPROACHES: A UNIFIED VIEW (5-6 pages)
   3.1 Frequentist Approaches
       - Sandwich estimators: the A⁻¹BA⁻¹ formula
       - When they work: sqrt(n)-consistency + CLT
       - When they fail: few clusters, leverage points, high dimensions
   3.2 Bootstrap Approaches
       - Pairs, wild, cluster, studentized
       - Connection to sandwich: bootstrap SE → sandwich SE
   3.3 Distribution-Free Approaches
       - Conformal: coverage from exchangeability alone
       - Universal inference: valid without regularity conditions
       - Limitation: typically prediction, not parameter inference
   3.4 Bayesian Approaches
       - Standard: concentrates at θ* but wrong variance
       - Generalized: loss-based posteriors with learning rate
       - The "temperature" puzzle: how to choose η?
   3.5 Game-Theoretic Approaches
       - E-values: valid under optional stopping
       - Safe testing under misspecification (largely open)

4. THEORETICAL RESULTS (5-7 pages)
   4.1 Unified Conditions for Validity
       - Theorem 1: Conditions under which sandwich CIs have correct coverage
       - Theorem 2: Conformal coverage guarantee (only exchangeability)
       - Theorem 3: Generalized Bayes calibration with optimal η
   4.2 The Misspecification-Efficiency Tradeoff
       - Theorem 4: No method can be simultaneously
         (a) fully efficient under correct specification
         (b) fully valid under arbitrary misspecification
       - Corollary: lower bound on the "robustness tax"
   4.3 The AMRI Method
       - Algorithm description
       - Theorem 5: AMRI achieves near-optimal efficiency under
         correct specification and valid coverage under misspecification
       - Proof sketch (full proof in supplement)
   4.4 Comparison of Theoretical Guarantees
       - Table summarizing what each method guarantees under each
         misspecification type

5. SIMULATION STUDY (6-8 pages)
   5.1 Design (with summary table of all DGPs)
   5.2 Results: Coverage Probability
       - Figure 1: Coverage vs. sample size, faceted by DGP and method
       - Key finding: which methods maintain 95% coverage?
   5.3 Results: Interval Width
       - Figure 2: Average width vs. sample size
       - Key finding: the efficiency cost of robustness
   5.4 Results: The Misspecification Spectrum
       - Figure 3: Coverage vs. misspecification severity (δ)
       - Key finding: where does each method break?
   5.5 Results: AMRI Performance
       - Figure 4: AMRI vs. best alternatives
   5.6 Summary and Recommendations
       - Table: when to use which method (practitioner guide)

6. REAL DATA APPLICATIONS (3-4 pages)
   6.1 Lalonde Data: Causal Inference Under Misspecification
   6.2 MEPS Data: Prediction Under Misspecification
   6.3 Key Findings

7. DISCUSSION (2-3 pages)
   7.1 Summary of Contributions
   7.2 Practical Recommendations for Applied Researchers
   7.3 Limitations
   7.4 Open Problems and Future Work
       - E-values under misspecification
       - Fairness guarantees under misspecification
       - High-dimensional misspecification theory
       - Connections to AI safety/alignment

REFERENCES (~100 references)

SUPPLEMENTARY MATERIAL (~25 pages)
   S1. Full Proofs of Theorems 1-5
   S2. Additional Simulation Results
       - All sample sizes × all DGPs × all methods
       - QQ-plots of z-scores
       - Monte Carlo standard errors
   S3. Additional Real Data: Riboflavin Genomics
   S4. Computational Details and Reproducibility
       - Code availability (GitHub link)
       - Runtime comparisons
   S5. Sensitivity to Simulation Design Choices
```

---

## 8. IMPLEMENTATION TIMELINE

### Phase 1: Literature Deep-Dive & Theory (Weeks 1-4)
- [ ] Complete reading of all foundational papers
- [ ] Develop formal taxonomy of misspecification types
- [ ] Prove theoretical results (Theorems 1-5)
- [ ] Design AMRI algorithm formally

### Phase 2: Simulation Infrastructure (Weeks 3-6)
- [ ] Set up simulation framework (SimDesign / custom Python)
- [ ] Implement all 6 DGPs with severity parameter
- [ ] Implement all 13 methods
- [ ] Pilot run: 1 DGP × 3 methods × 1000 reps (sanity check)

### Phase 3: Full Simulations (Weeks 5-8)
- [ ] Run full simulation: 6 DGPs × 6 sample sizes × 5 severity levels × 13 methods × 5000 reps
- [ ] Total: 6 × 6 × 5 × 13 × 5000 = 11,700,000 individual runs
- [ ] Parallel computing needed (Ray/cluster)
- [ ] Process results, create figures/tables

### Phase 4: Real Data Analysis (Weeks 7-9)
- [ ] Obtain and preprocess datasets (Lalonde, MEPS, Riboflavin)
- [ ] Run all methods on each dataset
- [ ] Interpret results

### Phase 5: Writing (Weeks 8-12)
- [ ] Draft each section
- [ ] Internal review and revision
- [ ] Prepare supplementary materials
- [ ] Code documentation and GitHub repository

### Phase 6: Submission (Week 12-13)
- [ ] Final proofreading
- [ ] Format for target journal
- [ ] Submit

---

## 9. KEY REFERENCES (Must-Read List)

### Foundational (Read First)
1. White, H. (1982). "Maximum likelihood estimation of misspecified models." *Econometrica*, 50(1), 1-25.
2. Buja, A. et al. (2019). "Models as Approximations I & II." *Statistical Science*, 34, 523-565.
3. Kleijn, B.J.K. & van der Vaart, A.W. (2012). "The BvM theorem under misspecification." *EJS*, 6, 354-381.
4. Barber, R.F. et al. (2021). "Predictive inference with the jackknife+." *Annals of Statistics*, 49, 486-507.
5. Chernozhukov, V. et al. (2018). "Double/debiased ML." *Econometrics Journal*, 21, C1-C68.

### Modern Methods
6. Bissiri, P.G., Holmes, C.C., & Walker, S.G. (2016). "General Bayesian updating." *JRSS-B*, 78, 1103-1130.
7. Grunwald, P.D. & van Ommen, T. (2017). "Inconsistency of Bayesian inference for misspecified linear models." *Bayesian Analysis*, 12, 1069-1103.
8. Miller, J.W. & Dunson, D.B. (2019). "Robust Bayesian inference via coarsening." *JASA*, 114, 1113-1125.
9. Wasserman, L., Ramdas, A., & Balakrishnan, S. (2020). "Universal inference." *PNAS*, 117, 16880-16890.
10. Romano, Y., Patterson, E., & Candes, E.J. (2019). "Conformalized quantile regression." *NeurIPS*.

### Game-Theoretic / E-Values
11. Grunwald, P.D., de Heide, R., & Koolen, W.M. (2024). "Safe testing." *JRSS-B*.
12. Ramdas, A. et al. (2023). "Game-theoretic statistics and safe anytime-valid inference." *Statistical Science*, 38(4), 576-601.
13. Fong, E. & Holmes, C.C. (2023). "Martingale posterior distributions." *JRSS-B*, 85, 1357-1391.

### Simulation Methodology
14. Morris, T.P., White, I.R., & Crowther, M.J. (2019). "Using simulation studies to evaluate statistical methods." *Statistics in Medicine*.

### Recent Cutting-Edge
15. Park, S., Balakrishnan, S., & Wasserman, L. (2023). "Robust Universal Inference for Misspecified Models." arXiv:2307.04034.
16. Xu et al. (2026). "Epistemic Traps." arXiv:2602.17676.
17. Young, A. & Shah, R. (2024). "Sandwich Regression." arXiv:2412.06119.
18. Wu et al. (2025). "DR variance failure." arXiv:2511.17907.

---

## 10. POTENTIAL IMPACT

### Academic Impact
- Bridges 4 major communities: robust statistics, Bayesian inference, distribution-free methods, causal inference
- First comprehensive empirical comparison of ALL major approaches
- Novel AMRI method addresses a fundamental open problem
- Clear practitioner guidelines (high citation potential)

### Practical Impact
- **Clinical trials**: Better uncertainty quantification when outcome models are wrong
- **Causal inference**: More reliable treatment effect estimates from observational data
- **ML deployment**: Formal guarantees for prediction intervals despite model misspecification
- **Fairness auditing**: Understanding when fairness guarantees break down
- **AI safety**: Connecting misspecification theory to alignment failures

### Target Venues (ranked)
1. **JASA (Theory & Methods)** — ideal for comprehensive theory + simulation paper
2. **Statistical Science** — ideal if framed as survey + new method
3. **JRSS-B** — ideal for theoretical depth with discussion format
4. **Biometrika** — if condensed to ~20 pages
5. **NeurIPS/ICML** — if emphasizing the ML/conformal/deep learning angle

---

## 11. RISK ASSESSMENT & MITIGATION

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Simulation takes too long | Medium | High | Use Ray for parallelization; start with pilot |
| Theoretical results too weak | Low-Medium | High | Focus on comprehensive empirics if theory stalls |
| Similar paper published during work | Medium | High | Focus on unification angle (unique contribution) |
| AMRI method doesn't outperform | Medium | Medium | Paper is valuable even as pure comparison study |
| Rejected from top venue | Medium | Low | Multiple suitable venues; revise and resubmit |

---

## 12. REPRODUCIBILITY PLAN

- **All code** on GitHub (MIT license)
- **All data** publicly available (Lalonde, MEPS, Riboflavin are all public)
- **Random seeds** documented for exact reproduction
- **Docker container** for computational environment
- **Makefile/targets pipeline** for one-command reproduction
- **Pre-registration** of simulation study design (optional but recommended)

---

*Document created: February 28, 2026*
*Research area: Mathematical Statistics / Machine Learning / Robust Inference*
*Keywords: model misspecification, robust inference, sandwich estimator, conformal prediction, generalized Bayes, universal inference, e-values*
