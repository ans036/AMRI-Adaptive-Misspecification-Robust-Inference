# References & Key Papers

A curated reading list of foundational and directly relevant papers for the AMRI research.
Papers are organized by topic with brief annotations on why each is important.

---

## 1. Robust Standard Errors (Sandwich Estimators)

The theoretical backbone of AMRI's robust mode.

| Paper | Why It Matters |
|-------|---------------|
| [White (1980). "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity." *Econometrica*, 48(4), 817–838.](https://doi.org/10.2307/1912934) | **Foundational.** Introduced HC0 — the sandwich estimator that remains consistent under heteroscedasticity. AMRI's SE ratio diagnostic is built on this. |
| [MacKinnon & White (1985). "Some Heteroskedasticity-Consistent Covariance Matrix Estimators with Improved Finite Sample Properties." *Journal of Econometrics*, 29(3), 305–325.](https://doi.org/10.1016/0304-4076(85)90158-7) | **Critical.** Introduced HC1, HC2, HC3. AMRI uses HC3 specifically because of its leverage correction and superior small-sample performance. |
| [Eicker (1967). "Limit Theorems for Regressions with Unequal and Dependent Errors." *Proceedings of the Fifth Berkeley Symposium*, 1, 59–82.](https://projecteuclid.org/euclid.bsmsp/1200512981) | Early derivation of heteroscedasticity-robust variance estimation. Predates White's formalization. |
| [Huber (1967). "The Behavior of Maximum Likelihood Estimates Under Nonstandard Conditions." *Proceedings of the Fifth Berkeley Symposium*, 1, 221–233.](https://projecteuclid.org/euclid.bsmsp/1200512988) | Introduced the "sandwich" structure (A⁻¹BA⁻¹) for variance estimation under misspecification. The M-estimation framework. |
| [Long & Ervin (2000). "Using Heteroscedasticity Consistent Standard Errors in the Linear Regression Model." *The American Statistician*, 54(3), 217–224.](https://doi.org/10.1080/00031305.2000.10474549) | Practical comparison of HC0–HC3. Shows HC3 is preferred for n < 250. Directly relevant to AMRI's choice of HC3. |
| [Cribari-Neto (2004). "Asymptotic Inference Under Heteroskedasticity of Unknown Form." *Computational Statistics & Data Analysis*, 45(2), 215–233.](https://doi.org/10.1016/S0167-9473(02)00366-3) | Introduced HC4 and HC5. Extensions we considered but HC3 sufficed. |
| [Angrist & Pischke (2009). *Mostly Harmless Econometrics*. Princeton University Press.](https://press.princeton.edu/books/paperback/9780691120355/mostly-harmless-econometrics) | Textbook treatment of when and why robust SEs matter. Chapter 8 is essential. |

---

## 2. Bootstrap Methods

The alternative robust approaches AMRI is benchmarked against.

| Paper | Why It Matters |
|-------|---------------|
| [Efron (1979). "Bootstrap Methods: Another Look at the Jackknife." *The Annals of Statistics*, 7(1), 1–26.](https://doi.org/10.1214/aos/1176344552) | **Foundational.** Introduced the bootstrap. Pairs bootstrap in our study is the direct application. |
| [Wu (1986). "Jackknife, Bootstrap and Other Resampling Methods in Regression Analysis." *The Annals of Statistics*, 14(4), 1261–1295.](https://doi.org/10.1214/aos/1176350142) | **Critical.** Introduced the wild bootstrap for regression. Shows it's consistent under heteroscedasticity. One of our 8 benchmark methods. |
| [Mammen (1993). "Bootstrap and Wild Bootstrap for High Dimensional Linear Models." *The Annals of Statistics*, 21(1), 255–285.](https://doi.org/10.1214/aos/1176349025) | Theoretical foundation for wild bootstrap. Shows Rademacher weights are optimal. We use Rademacher in our wild bootstrap implementation. |
| [Liu (1988). "Bootstrap Procedures Under Some Non-I.I.D. Models." *The Annals of Statistics*, 16(4), 1696–1708.](https://doi.org/10.1214/aos/1176351062) | Complements Wu (1986). Establishes conditions for wild bootstrap consistency. |
| [Hall (1992). *The Bootstrap and Edgeworth Expansion*. Springer.](https://doi.org/10.1007/978-1-4612-4384-7) | **Essential.** Theoretical foundation for the bootstrap-t (studentized bootstrap). Shows it achieves second-order accuracy — explaining why Boot-t has excellent coverage but very wide intervals in our results. |
| [DiCiccio & Efron (1996). "Bootstrap Confidence Intervals." *Statistical Science*, 11(3), 189–228.](https://doi.org/10.1214/ss/1032280214) | Comprehensive review of bootstrap CI methods (percentile, BCa, bootstrap-t). Contextualizes our comparison. |
| [Davison & Hinkley (1997). *Bootstrap Methods and Their Application*. Cambridge University Press.](https://doi.org/10.1017/CBO9780511802843) | Standard reference for applied bootstrap. Our implementation follows their algorithms. |
| [Flachaire (2005). "Bootstrapping Heteroskedastic Regression Models: Wild Bootstrap vs. Pairs Bootstrap." *Computational Statistics & Data Analysis*, 49(2), 361–376.](https://doi.org/10.1016/j.csda.2004.05.018) | Direct comparison of pairs vs wild bootstrap under heteroscedasticity. Explains differences we observe in our results. |

---

## 3. Model Misspecification Theory

The core problem AMRI addresses.

| Paper | Why It Matters |
|-------|---------------|
| [White (1982). "Maximum Likelihood Estimation of Misspecified Models." *Econometrica*, 50(1), 1–25.](https://doi.org/10.2307/1912526) | **Foundational.** Defines pseudo-true parameters (KL minimizers) and shows MLE converges to them under misspecification. AMRI targets pseudo-true parameters. |
| [Huber (1967). Cited above.](https://projecteuclid.org/euclid.bsmsp/1200512988) | M-estimation under misspecification. The sandwich variance is the correct variance even when the model is wrong. |
| [Berk (1966). "Limiting Behavior of Posterior Distributions When the Model Is Incorrect." *The Annals of Mathematical Statistics*, 37(1), 51–58.](https://doi.org/10.1214/aoms/1177699597) | Shows Bayesian posteriors concentrate on the KL-closest model even when wrong. Explains why Bayesian method fails in our simulations (same as Naive OLS). |
| [Kleijn & van der Vaart (2012). "The Bernstein-von Mises Theorem Under Misspecification." *Electronic Journal of Statistics*, 6, 354–381.](https://doi.org/10.1214/12-EJS675) | **Important.** The misspecified BvM theorem. Shows posterior is asymptotically normal but centered on the pseudo-true value with the WRONG variance. Directly explains our Bayesian coverage collapse. |
| [Müller (2013). "Risk of Bayesian Inference in Misspecified Models, and the Sandwich Covariance Matrix." *Econometrica*, 81(5), 1805–1849.](https://doi.org/10.3982/ECTA9097) | Connects Bayesian misspecification to sandwich estimation. Shows how to "fix" Bayesian inference — relevant to understanding why uncorrected Bayes fails. |
| [Lv & Liu (2014). "Model Selection Principles in Misspecified Models." *Journal of the Royal Statistical Society: Series B*, 76(1), 141–167.](https://doi.org/10.1111/rssb.12023) | Model selection under misspecification. Relevant to AMRI's adaptive switching. |

---

## 4. Adaptive & Pre-Test Inference

Most directly related to AMRI's switching mechanism.

| Paper | Why It Matters |
|-------|---------------|
| [Leeb & Pötscher (2005). "Model Selection and Inference: Facts and Fiction." *Econometric Theory*, 21(1), 21–59.](https://doi.org/10.1017/S0266466605050036) | **Critical warning.** Shows pre-test/model-selection estimators can have non-uniform risk. AMRI must address this — our threshold design specifically accounts for their concerns. |
| [Leeb & Pötscher (2006). "Can One Estimate the Conditional Distribution of Post-Model-Selection Estimators?" *The Annals of Statistics*, 34(5), 2554–2591.](https://doi.org/10.1214/009053606000000821) | Impossibility results for post-selection inference. AMRI avoids this by switching SEs (not models). |
| [Bauer, Pötscher & Hackl (1988). "Model Selection by Multiple Test Procedures." *Statistics*, 19(1), 39–44.](https://doi.org/10.1080/02331888808802068) | Early work on pre-test estimators. The concerns raised here motivated AMRI's conservative threshold. |
| [Andrews & Guggenberger (2009). "Hybrid and Size-Corrected Subsampling Methods." *Econometrica*, 77(3), 721–762.](https://doi.org/10.3982/ECTA7547) | Addresses non-uniform coverage in pre-test procedures. Their hybrid approach inspired AMRI's continuous-threshold idea. |
| [McCloskey (2017). "Bonferroni-Based Size-Correction for Nonstandard Testing Problems." *Journal of Econometrics*, 200(1), 17–35.](https://doi.org/10.1016/j.jeconom.2017.05.001) | Size correction for adaptive procedures. Relevant to AMRI's 1.05 inflation factor. |
| [Romano & Wolf (2005). "Stepwise Multiple Testing as Formalized Data Snooping." *Econometrica*, 73(4), 1237–1282.](https://doi.org/10.1111/j.1468-0262.2005.00615.x) | Multiple testing framework. Relevant to our Hochberg corrections in hypothesis testing. |

---

## 5. Misspecification Detection & Diagnostics

Papers on detecting when models are wrong — AMRI's SE ratio diagnostic.

| Paper | Why It Matters |
|-------|---------------|
| [Hausman (1978). "Specification Tests in Econometrics." *Econometrica*, 46(6), 1251–1271.](https://doi.org/10.2307/1913827) | **Foundational.** The Hausman test compares efficient vs consistent estimators — exactly AMRI's SE ratio logic. AMRI can be viewed as a Hausman-type diagnostic applied to variance estimation. |
| [White (1980). Cited above — Section 3.](https://doi.org/10.2307/1912934) | Includes the "information matrix test" for misspecification detection. |
| [Ramsey (1969). "Tests for Specification Errors in Classical Linear Least-Squares Regression Analysis." *Journal of the Royal Statistical Society: Series B*, 31(2), 350–371.](https://doi.org/10.1111/j.2517-6161.1969.tb00796.x) | RESET test for functional form misspecification. An alternative diagnostic that AMRI's SE ratio implicitly captures. |
| [Cook & Weisberg (1983). "Diagnostics for Heteroscedasticity in Regression." *Biometrika*, 70(1), 1–10.](https://doi.org/10.1093/biomet/70.1.1) | Score test for heteroscedasticity. AMRI's SE ratio captures this signal without explicitly testing. |

---

## 6. Confidence Interval Theory

Foundational coverage probability theory.

| Paper | Why It Matters |
|-------|---------------|
| [Neyman (1937). "Outline of a Theory of Statistical Estimation Based on the Classical Theory of Probability." *Philosophical Transactions of the Royal Society A*, 236(767), 333–380.](https://doi.org/10.1098/rsta.1937.0005) | **Foundational.** Invented confidence intervals. Defines the coverage probability guarantee AMRI aims to maintain. |
| [Brown, Cai & DasGupta (2001). "Interval Estimation for a Binomial Proportion." *Statistical Science*, 16(2), 101–133.](https://doi.org/10.1214/ss/1009213286) | Wilson intervals for coverage estimation. We use Wilson CIs to assess our simulation coverage. |
| [Blyth & Still (1983). "Binomial Confidence Intervals." *Journal of the American Statistical Association*, 78(381), 108–116.](https://doi.org/10.1080/01621459.1983.10477938) | Exact binomial CIs. Relevant to interpreting our Monte Carlo coverage estimates. |

---

## 7. Statistical Testing Methodology

Methods used in our formal hypothesis testing framework.

| Paper | Why It Matters |
|-------|---------------|
| [Fisher (1932). *Statistical Methods for Research Workers*. Oliver & Boyd.](https://doi.org/10.1007/978-1-4612-4380-9_6) | Fisher's combined probability test. We use it for combining p-values across DGPs in Test 1. |
| [Hochberg (1988). "A Sharper Bonferroni Procedure for Multiple Tests of Significance." *Biometrika*, 75(4), 800–802.](https://doi.org/10.1093/biomet/75.4.800) | **Used directly.** Hochberg step-up procedure for FWER control. We apply it to test all 5 hypotheses simultaneously. |
| [Simes (1986). "An Improved Bonferroni Procedure for Multiple Tests of Significance." *Biometrika*, 73(3), 751–754.](https://doi.org/10.1093/biomet/73.3.751) | **Used directly.** Simes test for intersection null. We use it for H2 (HC3 universal robustness). |
| [Schuirmann (1987). "A Comparison of the Two One-Sided Tests Procedure and the Power Approach for Assessing the Equivalence of Average Bioavailability." *Journal of Pharmacokinetics and Biopharmaceutics*, 15(6), 657–680.](https://doi.org/10.1007/BF01068419) | **Used directly.** TOST equivalence testing. We use it to formally prove AMRI coverage is equivalent to 0.95. |
| [Lakens (2017). "Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses." *Social Psychological and Personality Science*, 8(4), 355–362.](https://doi.org/10.1177/1948550617697177) | Modern treatment of TOST. Guided our epsilon choices (0.01, 0.02, 0.03). |
| [Cohen (1988). *Statistical Power Analysis for the Behavioral Sciences*. Routledge.](https://doi.org/10.4324/9780203771587) | Cohen's d effect size benchmarks (small=0.2, medium=0.5, large=0.8). We report effect sizes for all AMRI comparisons. |
| [Mann & Whitney (1947). "On a Test of Whether One of Two Random Variables is Stochastically Larger than the Other." *The Annals of Mathematical Statistics*, 18(1), 50–60.](https://doi.org/10.1214/aoms/1177730491) | **Used directly.** Mann-Whitney U test for degradation rate ordering (H4). |
| [Good (2005). *Permutation, Parametric and Bootstrap Tests of Hypotheses*. Springer.](https://doi.org/10.1007/b138696) | Reference for our distribution-free permutation test (50K reps) confirming AMRI > HC3. |

---

## 8. Bayesian Inference Under Misspecification

Why Bayesian methods fail in our simulations.

| Paper | Why It Matters |
|-------|---------------|
| [Kleijn & van der Vaart (2012). Cited above.](https://doi.org/10.1214/12-EJS675) | Misspecified BvM theorem — posterior converges to wrong variance. |
| [Grünwald & van Ommen (2017). "Inconsistency of Bayesian Inference for Misspecified Linear Bayesian Networks." *UAI*.](https://proceedings.mlr.press/v62/grünwald17a.html) | Shows Bayesian inconsistency under misspecification. |
| [Müller (2013). Cited above.](https://doi.org/10.3982/ECTA9097) | The connection between Bayesian risk and sandwich covariance. |
| [Syring & Martin (2019). "Calibrating General Posterior Credible Regions." *Biometrika*, 106(2), 479–486.](https://doi.org/10.1093/biomet/asy054) | Methods to recalibrate Bayesian CIs under misspecification. A potential future extension for AMRI. |

---

## 9. Sample Size Paradox & Asymptotics

The counterintuitive finding that more data hurts naive inference.

| Paper | Why It Matters |
|-------|---------------|
| [White (1982). Cited above.](https://doi.org/10.2307/1912526) | Pseudo-true parameter theory explains why OLS converges to a wrong value — and why the CI shrinks around it. |
| [Freedman (2006). "On the So-Called 'Huber Sandwich Estimator' and 'Robust Standard Errors'." *The American Statistician*, 60(4), 299–302.](https://doi.org/10.1198/000313006X152207) | **Important critique.** Warns that sandwich SEs are not a panacea. The point estimate may still converge to the wrong thing. AMRI inherits this limitation (acknowledged in our paper). |
| [King & Roberts (2015). "How Robust Standard Errors Expose Methodological Problems They Do Not Fix, and What to Do About It." *Political Analysis*, 23(2), 159–179.](https://doi.org/10.1093/pan/mpu015) | Argues robust SEs are a band-aid. Relevant to our discussion of AMRI's scope and limitations. |

---

## 10. Simulation Methodology

How we designed and validated our Monte Carlo study.

| Paper | Why It Matters |
|-------|---------------|
| [Burton, Altman, Royston & Holder (2006). "The Design of Simulation Studies in Medical Statistics." *Statistics in Medicine*, 25(24), 4279–4292.](https://doi.org/10.1002/sim.2673) | Best practices for simulation design. Our 2000 reps, 6 DGPs, and factorial design follow their guidelines. |
| [Morris, White & Crowther (2019). "Using Simulation Studies to Evaluate Statistical Methods." *Statistics in Medicine*, 38(11), 2074–2102.](https://doi.org/10.1002/sim.8086) | **Essential.** Modern framework for simulation studies. Influenced our reporting of Monte Carlo SE, performance measures, and DGP selection. |
| [Boulesteix, Groenwold, Abrahamowicz et al. (2020). "Introduction to Statistical Simulations in Health Research." *BMJ Open*, 10, e039921.](https://doi.org/10.1136/bmjopen-2020-039921) | Guidelines for reporting simulation studies. Our HYPOTHESES.md follows their recommended structure. |

---

## Reading Priority

### Must Read (Core Theory)
1. White (1980) — HC0 sandwich estimator
2. MacKinnon & White (1985) — HC3
3. White (1982) — Misspecification theory
4. Hausman (1978) — Specification tests (AMRI's intellectual ancestor)
5. Leeb & Pötscher (2005) — Pre-test inference risks
6. Kleijn & van der Vaart (2012) — Bayesian misspecification

### Should Read (Methods We Use)
7. Wu (1986) — Wild bootstrap
8. Hall (1992) — Bootstrap-t theory
9. Hochberg (1988) — Multiple testing
10. Schuirmann (1987) — TOST equivalence
11. Morris et al. (2019) — Simulation methodology

### Good to Read (Context & Extensions)
12. Freedman (2006) — Sandwich SE limitations
13. King & Roberts (2015) — Robust SEs critique
14. Müller (2013) — Bayesian + sandwich connection
15. Long & Ervin (2000) — Practical HC comparison
16. Angrist & Pischke (2009) — Applied perspective

---

## How These Papers Connect to AMRI

```
White (1980) ──────── HC0 sandwich estimator
       │
MacKinnon & White (1985) ── HC3 (what AMRI uses)
       │
Hausman (1978) ──────── Compare efficient vs consistent
       │                     │
       │              AMRI's SE ratio = SE_HC3 / SE_naive
       │                     │
       │              Leeb & Pötscher (2005) ── Pre-test risks
       │                     │
       │              AMRI's threshold τ(n) = 1 + 2/√n
       │              (adapts to sampling noise)
       │
White (1982) ────────── Pseudo-true parameters
       │                     │
       │              Freedman (2006) ── Sandwich limitations
       │                     │
       │              AMRI targets pseudo-true β*
       │              (honest about what it estimates)
       │
Kleijn & van der Vaart (2012) ── Why Bayesian fails
       │
Wu (1986) + Hall (1992) ──── Bootstrap alternatives
       │
Morris et al. (2019) ──── How we evaluate everything
```

---

*Last updated: February 2026*
