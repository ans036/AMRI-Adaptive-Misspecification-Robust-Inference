"""
REAL DATA VALIDATION
=====================
Test AMRI on ACTUAL real-world datasets (not simulated).

Uses:
- sklearn: Boston Housing (via fetch_openml), California Housing, Diabetes
- statsmodels: CPS earnings, Fair wage, Longley, Duncan prestige
- rdatasets: Lalonde (NSW), Auto (MPG), mtcars, Prestige, Affairs

For each dataset:
1. Fit a simple linear regression on a natural (X, Y) pair
2. Compute Naive SE, HC3 SE, and AMRI SE
3. Run a large-scale bootstrap (B=50000) to get the "true" sampling distribution
4. Compare each method's SE to the bootstrap SE (ground truth)
5. Compute bootstrap coverage: how often does each CI capture the bootstrap mean?
"""
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def ols_analysis(X, Y, dataset_name, var_names):
    """Run full OLS analysis with Naive, HC3, and AMRI."""
    n = len(X)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    # Remove NaN
    mask = ~(np.isnan(X) | np.isnan(Y))
    X, Y = X[mask], Y[mask]
    n = len(X)

    if n < 30:
        print(f"  {dataset_name}: SKIPPED (n={n} < 30)")
        return None

    # OLS
    Xc = X - X.mean()
    Yc = Y - Y.mean()
    SXX = (Xc**2).sum()
    slope = (Xc * Yc).sum() / SXX
    intercept = Y.mean() - slope * X.mean()
    resid = Y - intercept - slope * X
    sigma2 = (resid**2).sum() / (n - 2)

    # Naive SE
    se_naive = np.sqrt(sigma2 / SXX)

    # HC3 SE
    h = 1/n + Xc**2 / SXX
    meat = (Xc**2 * resid**2 / (1 - h)**2).sum()
    se_hc3 = np.sqrt(meat / SXX**2)

    # AMRI
    ratio = se_hc3 / max(se_naive, 1e-10)
    threshold = 1 + 2 / np.sqrt(n)
    if ratio > threshold or ratio < 1/threshold:
        se_amri = se_hc3 * 1.05
        mode = "ROBUST"
    else:
        se_amri = se_naive
        mode = "EFFICIENT"

    t_crit = stats.t.ppf(0.975, n - 2)

    # Bootstrap ground truth (50000 reps)
    rng = np.random.default_rng(42)
    B = 50000
    boot_slopes = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        Xb, Yb = X[idx], Y[idx]
        Xbc = Xb - Xb.mean()
        Ybc = Yb - Yb.mean()
        denom = (Xbc**2).sum()
        if denom < 1e-10:
            boot_slopes[b] = slope
        else:
            boot_slopes[b] = (Xbc * Ybc).sum() / denom
    true_se = np.std(boot_slopes)

    # Bootstrap coverage: what fraction of bootstrap CIs would capture
    # the "true" slope (using bootstrap distribution as ground truth)
    # We test: for each method's SE, how often does slope +/- t*SE contain boot_slope?
    naive_covers = ((slope - t_crit * se_naive <= boot_slopes) &
                    (boot_slopes <= slope + t_crit * se_naive)).mean()
    hc3_covers = ((slope - t_crit * se_hc3 <= boot_slopes) &
                  (boot_slopes <= slope + t_crit * se_hc3)).mean()
    amri_covers = ((slope - t_crit * se_amri <= boot_slopes) &
                   (boot_slopes <= slope + t_crit * se_amri)).mean()

    # SE errors
    naive_err = abs(se_naive - true_se) / true_se * 100
    hc3_err = abs(se_hc3 - true_se) / true_se * 100
    amri_err = abs(se_amri - true_se) / true_se * 100

    result = {
        'dataset': dataset_name,
        'variables': f'{var_names[0]} -> {var_names[1]}',
        'n': n,
        'slope': slope,
        'se_ratio': ratio,
        'amri_mode': mode,
        'se_naive': se_naive, 'se_hc3': se_hc3, 'se_amri': se_amri,
        'true_se_bootstrap': true_se,
        'naive_se_error_pct': naive_err,
        'hc3_se_error_pct': hc3_err,
        'amri_se_error_pct': amri_err,
        'naive_boot_coverage': naive_covers,
        'hc3_boot_coverage': hc3_covers,
        'amri_boot_coverage': amri_covers,
    }

    print(f"  {dataset_name} ({var_names[0]}->{var_names[1]}, n={n})")
    print(f"    SE ratio={ratio:.3f}, threshold={threshold:.3f}, AMRI mode={mode}")
    print(f"    True SE (bootstrap 50K): {true_se:.6f}")
    print(f"    Naive SE: {se_naive:.6f} (err={naive_err:.1f}%), boot_cov={naive_covers:.4f}")
    print(f"    HC3 SE:   {se_hc3:.6f} (err={hc3_err:.1f}%), boot_cov={hc3_covers:.4f}")
    print(f"    AMRI SE:  {se_amri:.6f} (err={amri_err:.1f}%), boot_cov={amri_covers:.4f}")

    # Who's closest to 0.95 coverage?
    diffs = {
        'Naive': abs(naive_covers - 0.95),
        'HC3': abs(hc3_covers - 0.95),
        'AMRI': abs(amri_covers - 0.95),
    }
    best = min(diffs, key=diffs.get)
    print(f"    Closest to 0.95 coverage: {best} ({diffs[best]:.4f} away)")
    print()

    return result


def load_real_datasets():
    """Load actual real-world datasets from various packages."""
    datasets = []

    # === SKLEARN DATASETS ===
    print("Loading sklearn datasets...")
    try:
        from sklearn.datasets import fetch_california_housing
        cal = fetch_california_housing()
        df_cal = pd.DataFrame(cal.data, columns=cal.feature_names)
        df_cal['price'] = cal.target
        datasets.append(('California_Housing', df_cal['MedInc'].values, df_cal['price'].values,
                         ('MedInc', 'HousePrice')))
        datasets.append(('California_Rooms', df_cal['AveRooms'].values, df_cal['price'].values,
                         ('AveRooms', 'HousePrice')))
    except Exception as e:
        print(f"  California Housing failed: {e}")

    try:
        from sklearn.datasets import load_diabetes
        diab = load_diabetes()
        df_diab = pd.DataFrame(diab.data, columns=diab.feature_names)
        df_diab['target'] = diab.target
        datasets.append(('Diabetes_BMI', df_diab['bmi'].values, df_diab['target'].values,
                         ('BMI', 'DiseaseProgress')))
        datasets.append(('Diabetes_BP', df_diab['bp'].values, df_diab['target'].values,
                         ('BloodPressure', 'DiseaseProgress')))
    except Exception as e:
        print(f"  Diabetes failed: {e}")

    # === STATSMODELS DATASETS ===
    print("Loading statsmodels datasets...")
    try:
        import statsmodels.api as sm
        duncan = sm.datasets.get_rdataset('Duncan', 'carData').data
        datasets.append(('Duncan_Prestige', duncan['income'].values, duncan['prestige'].values,
                         ('Income', 'Prestige')))
    except Exception as e:
        print(f"  Duncan failed: {e}")

    try:
        import statsmodels.api as sm
        longley = sm.datasets.longley.load_pandas().data
        datasets.append(('Longley_Employment', longley['GNP'].values, longley['TOTEMP'].values,
                         ('GNP', 'TotalEmployment')))
    except Exception as e:
        print(f"  Longley failed: {e}")

    try:
        import statsmodels.api as sm
        fair = sm.datasets.fair.load_pandas().data
        datasets.append(('Fair_Affairs', fair['rate_marriage'].values, fair['affairs'].values,
                         ('MarriageRating', 'Affairs')))
        datasets.append(('Fair_Age', fair['age'].values, fair['affairs'].values,
                         ('Age', 'Affairs')))
    except Exception as e:
        print(f"  Fair failed: {e}")

    try:
        import statsmodels.api as sm
        star98 = sm.datasets.star98.load_pandas().data
        datasets.append(('Star98_Math', star98.iloc[:, 1].values, star98.iloc[:, 0].values,
                         ('Feature1', 'MathScore')))
    except Exception as e:
        print(f"  Star98 failed: {e}")

    # === RDATASETS ===
    print("Loading rdatasets...")
    try:
        from rdatasets import data as rdata

        # mtcars
        mtcars = rdata('mtcars')
        datasets.append(('mtcars_HP', mtcars['hp'].values, mtcars['mpg'].values,
                         ('Horsepower', 'MPG')))
        datasets.append(('mtcars_Weight', mtcars['wt'].values, mtcars['mpg'].values,
                         ('Weight', 'MPG')))

        # iris
        iris = rdata('iris')
        datasets.append(('Iris_Sepal', iris['Sepal.Length'].values, iris['Sepal.Width'].values,
                         ('SepalLength', 'SepalWidth')))

        # Auto (ISLR)
        try:
            auto = rdata('Auto', 'ISLR')
            datasets.append(('Auto_HP', auto['horsepower'].values.astype(float),
                             auto['mpg'].values.astype(float), ('Horsepower', 'MPG')))
            datasets.append(('Auto_Weight', auto['weight'].values.astype(float),
                             auto['mpg'].values.astype(float), ('Weight', 'MPG')))
        except:
            pass

        # Prestige
        try:
            prestige = rdata('Prestige', 'carData')
            datasets.append(('Prestige_Income', prestige['income'].values,
                             prestige['prestige'].values, ('Income', 'Prestige')))
            datasets.append(('Prestige_Education', prestige['education'].values,
                             prestige['prestige'].values, ('Education', 'Prestige')))
        except:
            pass

        # Boston (MASS)
        try:
            boston = rdata('Boston', 'MASS')
            datasets.append(('Boston_Rooms', boston['rm'].values, boston['medv'].values,
                             ('AvgRooms', 'MedianValue')))
            datasets.append(('Boston_LSTAT', boston['lstat'].values, boston['medv'].values,
                             ('LowerStatus%', 'MedianValue')))
            datasets.append(('Boston_Crime', boston['crim'].values, boston['medv'].values,
                             ('CrimeRate', 'MedianValue')))
        except:
            pass

        # Wage (ISLR)
        try:
            wage = rdata('Wage', 'ISLR')
            datasets.append(('Wage_Age', wage['age'].values.astype(float),
                             wage['wage'].values.astype(float), ('Age', 'Wage')))
            datasets.append(('Wage_Education', wage['education'].cat.codes.values.astype(float),
                             wage['wage'].values.astype(float), ('EducLevel', 'Wage')))
        except:
            pass

        # Lalonde (Matching)
        try:
            lalonde = rdata('lalonde', 'Matching')
            datasets.append(('Lalonde_Treat', lalonde['treat'].values.astype(float),
                             lalonde['re78'].values.astype(float), ('Treatment', 'Earnings78')))
            datasets.append(('Lalonde_Age', lalonde['age'].values.astype(float),
                             lalonde['re78'].values.astype(float), ('Age', 'Earnings78')))
            datasets.append(('Lalonde_Educ', lalonde['educ'].values.astype(float),
                             lalonde['re78'].values.astype(float), ('Education', 'Earnings78')))
        except:
            pass

        # diamonds (ggplot2)
        try:
            diamonds = rdata('diamonds', 'ggplot2')
            # Subsample for speed
            idx = np.random.default_rng(42).choice(len(diamonds), 2000, replace=False)
            d_sub = diamonds.iloc[idx]
            datasets.append(('Diamonds_Carat', d_sub['carat'].values.astype(float),
                             d_sub['price'].values.astype(float), ('Carat', 'Price')))
        except:
            pass

    except ImportError:
        print("  rdatasets not available")
    except Exception as e:
        print(f"  rdatasets error: {e}")

    print(f"\nLoaded {len(datasets)} real-world dataset pairs.\n")
    return datasets


if __name__ == '__main__':
    print("=" * 80)
    print("REAL DATA VALIDATION: AMRI on ACTUAL Real-World Datasets")
    print("=" * 80)
    print("Bootstrap ground truth: 50,000 resamples per dataset\n")

    datasets = load_real_datasets()

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    results = []
    for name, X, Y, var_names in datasets:
        res = ols_analysis(X, Y, name, var_names)
        if res is not None:
            results.append(res)

    if not results:
        print("No results!")
        exit(1)

    res_df = pd.DataFrame(results)

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # How often is AMRI in the right mode?
    n_total = len(res_df)
    n_robust = (res_df['amri_mode'] == 'ROBUST').sum()
    n_efficient = (res_df['amri_mode'] == 'EFFICIENT').sum()
    print(f"\n  Total datasets: {n_total}")
    print(f"  AMRI chose ROBUST: {n_robust} ({n_robust/n_total*100:.0f}%)")
    print(f"  AMRI chose EFFICIENT: {n_efficient} ({n_efficient/n_total*100:.0f}%)")

    # Average SE errors
    print(f"\n  Average SE error (% from bootstrap truth):")
    print(f"    Naive: {res_df['naive_se_error_pct'].mean():.1f}%")
    print(f"    HC3:   {res_df['hc3_se_error_pct'].mean():.1f}%")
    print(f"    AMRI:  {res_df['amri_se_error_pct'].mean():.1f}%")

    # Average bootstrap coverage
    print(f"\n  Average bootstrap coverage (target: 0.95):")
    print(f"    Naive: {res_df['naive_boot_coverage'].mean():.4f}")
    print(f"    HC3:   {res_df['hc3_boot_coverage'].mean():.4f}")
    print(f"    AMRI:  {res_df['amri_boot_coverage'].mean():.4f}")

    # Who's closest to 0.95?
    res_df['naive_dist_95'] = abs(res_df['naive_boot_coverage'] - 0.95)
    res_df['hc3_dist_95'] = abs(res_df['hc3_boot_coverage'] - 0.95)
    res_df['amri_dist_95'] = abs(res_df['amri_boot_coverage'] - 0.95)

    amri_closest = (res_df['amri_dist_95'] <= res_df[['naive_dist_95', 'hc3_dist_95']].min(axis=1) + 0.005).sum()
    naive_closest = (res_df['naive_dist_95'] <= res_df[['amri_dist_95', 'hc3_dist_95']].min(axis=1) + 0.005).sum()
    hc3_closest = (res_df['hc3_dist_95'] <= res_df[['naive_dist_95', 'amri_dist_95']].min(axis=1) + 0.005).sum()

    print(f"\n  Closest to 0.95 coverage (within 0.005):")
    print(f"    AMRI:  {amri_closest}/{n_total} datasets")
    print(f"    HC3:   {hc3_closest}/{n_total} datasets")
    print(f"    Naive: {naive_closest}/{n_total} datasets")

    # Datasets where misspecification matters (ratio > 1.1 or < 0.9)
    misspec = res_df[res_df['se_ratio'] > 1.1]
    if len(misspec) > 0:
        print(f"\n  Datasets with detected misspecification (SE ratio > 1.1): {len(misspec)}")
        for _, row in misspec.iterrows():
            print(f"    {row['dataset']}: ratio={row['se_ratio']:.3f}, "
                  f"Naive coverage={row['naive_boot_coverage']:.4f}, "
                  f"AMRI coverage={row['amri_boot_coverage']:.4f}")

    # Statistical test: is AMRI coverage closer to 0.95 than Naive?
    print(f"\n  Paired test: AMRI vs Naive distance from 0.95")
    diffs = res_df['naive_dist_95'] - res_df['amri_dist_95']
    t_stat, p_val = stats.ttest_1samp(diffs, 0)
    print(f"    Mean diff: {diffs.mean():.5f} (positive = AMRI closer to 0.95)")
    print(f"    t={t_stat:.3f}, p={p_val:.4f}")
    print(f"    Result: {'AMRI significantly closer to 0.95' if p_val < 0.05 and diffs.mean() > 0 else 'Not significant'}")

    # Save
    figs = Path("c:/Users/anish/OneDrive/Desktop/Novel Research/figures")
    res_df.to_csv(figs / 'real_data_results.csv', index=False)
    print(f"\n  Results saved to figures/real_data_results.csv")
    print("\nDONE.")
