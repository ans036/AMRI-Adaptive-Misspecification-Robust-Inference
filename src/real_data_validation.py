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
from pathlib import Path
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

    # AMRI v1 (hard switching)
    ratio = se_hc3 / max(se_naive, 1e-10)
    threshold = 1 + 2 / np.sqrt(n)
    if ratio > threshold or ratio < 1/threshold:
        se_amri = se_hc3 * 1.05
        mode = "ROBUST"
    else:
        se_amri = se_naive
        mode = "EFFICIENT"

    # AMRI v2 (soft thresholding)
    c1, c2 = 1.0, 2.0
    log_ratio = abs(np.log(ratio))
    lower_t = c1 / np.sqrt(n)
    upper_t = c2 / np.sqrt(n)
    if upper_t > lower_t:
        w = np.clip((log_ratio - lower_t) / (upper_t - lower_t), 0.0, 1.0)
    else:
        w = 1.0 if log_ratio > lower_t else 0.0
    se_amri_v2 = (1 - w) * se_naive + w * se_hc3

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
    amri_v2_covers = ((slope - t_crit * se_amri_v2 <= boot_slopes) &
                      (boot_slopes <= slope + t_crit * se_amri_v2)).mean()

    # SE errors
    naive_err = abs(se_naive - true_se) / true_se * 100
    hc3_err = abs(se_hc3 - true_se) / true_se * 100
    amri_err = abs(se_amri - true_se) / true_se * 100
    amri_v2_err = abs(se_amri_v2 - true_se) / true_se * 100

    result = {
        'dataset': dataset_name,
        'variables': f'{var_names[0]} -> {var_names[1]}',
        'n': n,
        'slope': slope,
        'se_ratio': ratio,
        'amri_mode': mode,
        'amri_v2_weight': w,
        'se_naive': se_naive, 'se_hc3': se_hc3, 'se_amri': se_amri, 'se_amri_v2': se_amri_v2,
        'true_se_bootstrap': true_se,
        'naive_se_error_pct': naive_err,
        'hc3_se_error_pct': hc3_err,
        'amri_se_error_pct': amri_err,
        'amri_v2_se_error_pct': amri_v2_err,
        'naive_boot_coverage': naive_covers,
        'hc3_boot_coverage': hc3_covers,
        'amri_boot_coverage': amri_covers,
        'amri_v2_boot_coverage': amri_v2_covers,
    }

    print(f"  {dataset_name} ({var_names[0]}->{var_names[1]}, n={n})")
    print(f"    SE ratio={ratio:.3f}, threshold={threshold:.3f}, AMRI v1={mode}, v2 w={w:.3f}")
    print(f"    True SE (bootstrap 50K): {true_se:.6f}")
    print(f"    Naive SE:    {se_naive:.6f} (err={naive_err:.1f}%), boot_cov={naive_covers:.4f}")
    print(f"    HC3 SE:      {se_hc3:.6f} (err={hc3_err:.1f}%), boot_cov={hc3_covers:.4f}")
    print(f"    AMRI v1 SE:  {se_amri:.6f} (err={amri_err:.1f}%), boot_cov={amri_covers:.4f}")
    print(f"    AMRI v2 SE:  {se_amri_v2:.6f} (err={amri_v2_err:.1f}%), boot_cov={amri_v2_covers:.4f}")

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


def _try_add(datasets, name, X, Y, var_names):
    """Safely add a dataset, converting to numeric and dropping NaN."""
    try:
        X = pd.to_numeric(pd.Series(X), errors='coerce').values.astype(float)
        Y = pd.to_numeric(pd.Series(Y), errors='coerce').values.astype(float)
        mask = ~(np.isnan(X) | np.isnan(Y))
        X, Y = X[mask], Y[mask]
        if len(X) >= 20:
            datasets.append((name, X, Y, var_names))
    except Exception:
        pass


def load_real_datasets():
    """Load 30+ real-world datasets from sklearn, statsmodels, rdatasets, and OpenML."""
    datasets = []

    # === SKLEARN DATASETS ===
    print("Loading sklearn datasets...")
    try:
        from sklearn.datasets import fetch_california_housing
        cal = fetch_california_housing()
        df_cal = pd.DataFrame(cal.data, columns=cal.feature_names)
        df_cal['price'] = cal.target
        _try_add(datasets, 'California_Housing', df_cal['MedInc'], df_cal['price'],
                 ('MedInc', 'HousePrice'))
        _try_add(datasets, 'California_Rooms', df_cal['AveRooms'], df_cal['price'],
                 ('AveRooms', 'HousePrice'))
        _try_add(datasets, 'California_Age', df_cal['HouseAge'], df_cal['price'],
                 ('HouseAge', 'HousePrice'))
    except Exception as e:
        print(f"  California Housing failed: {e}")

    try:
        from sklearn.datasets import load_diabetes
        diab = load_diabetes()
        df_diab = pd.DataFrame(diab.data, columns=diab.feature_names)
        df_diab['target'] = diab.target
        _try_add(datasets, 'Diabetes_BMI', df_diab['bmi'], df_diab['target'],
                 ('BMI', 'DiseaseProgress'))
        _try_add(datasets, 'Diabetes_BP', df_diab['bp'], df_diab['target'],
                 ('BloodPressure', 'DiseaseProgress'))
        _try_add(datasets, 'Diabetes_S5', df_diab['s5'], df_diab['target'],
                 ('S5_ltg', 'DiseaseProgress'))
    except Exception as e:
        print(f"  Diabetes failed: {e}")

    # === STATSMODELS DATASETS ===
    print("Loading statsmodels datasets...")
    try:
        import statsmodels.api as sm
        duncan = sm.datasets.get_rdataset('Duncan', 'carData').data
        _try_add(datasets, 'Duncan_Prestige', duncan['income'], duncan['prestige'],
                 ('Income', 'Prestige'))
    except Exception as e:
        print(f"  Duncan failed: {e}")

    try:
        import statsmodels.api as sm
        fair = sm.datasets.fair.load_pandas().data
        _try_add(datasets, 'Fair_Affairs', fair['rate_marriage'], fair['affairs'],
                 ('MarriageRating', 'Affairs'))
        _try_add(datasets, 'Fair_Age', fair['age'], fair['affairs'],
                 ('Age', 'Affairs'))
    except Exception as e:
        print(f"  Fair failed: {e}")

    try:
        import statsmodels.api as sm
        star98 = sm.datasets.star98.load_pandas().data
        _try_add(datasets, 'Star98_Math', star98.iloc[:, 1], star98.iloc[:, 0],
                 ('Feature1', 'MathScore'))
    except Exception as e:
        print(f"  Star98 failed: {e}")

    # === OPENML DATASETS ===
    print("Loading OpenML datasets...")
    try:
        from sklearn.datasets import fetch_openml

        try:
            ab = fetch_openml(name='abalone', version=1, as_frame=True, parser='auto')
            df_ab = ab.frame
            _try_add(datasets, 'Abalone_Length', df_ab['Length'], df_ab['Class_number_of_rings'],
                     ('Length', 'Rings'))
            _try_add(datasets, 'Abalone_ShellWt', df_ab['Shell_weight'], df_ab['Class_number_of_rings'],
                     ('ShellWeight', 'Rings'))
            _try_add(datasets, 'Abalone_Diameter', df_ab['Diameter'], df_ab['Class_number_of_rings'],
                     ('Diameter', 'Rings'))
        except Exception as e:
            print(f"  Abalone failed: {e}")

        try:
            am = fetch_openml(name='autoMpg', version=1, as_frame=True, parser='auto')
            df_mpg = am.frame
            _try_add(datasets, 'AutoMPG_HP', df_mpg['horsepower'], df_mpg['class'],
                     ('Horsepower', 'MPG'))
            _try_add(datasets, 'AutoMPG_Weight', df_mpg['weight'], df_mpg['class'],
                     ('Weight', 'MPG'))
        except Exception as e:
            print(f"  Auto MPG failed: {e}")

        try:
            wine = fetch_openml(name='wine-quality-red', version=1, as_frame=True, parser='auto')
            df_wine = wine.frame
            _try_add(datasets, 'Wine_Alcohol', df_wine['alcohol'], df_wine['class'],
                     ('Alcohol', 'Quality'))
            _try_add(datasets, 'Wine_Acidity', df_wine['volatile_acidity'], df_wine['class'],
                     ('VolatileAcidity', 'Quality'))
        except Exception as e:
            print(f"  Wine Quality failed: {e}")

    except ImportError:
        print("  fetch_openml not available")

    # === RDATASETS (package, item) API ===
    print("Loading rdatasets...")
    try:
        from rdatasets import data as rdata

        # --- Base R datasets ---
        try:
            mtcars = rdata('datasets', 'mtcars')
            _try_add(datasets, 'mtcars_HP', mtcars['hp'], mtcars['mpg'], ('Horsepower', 'MPG'))
            _try_add(datasets, 'mtcars_Weight', mtcars['wt'], mtcars['mpg'], ('Weight', 'MPG'))
            _try_add(datasets, 'mtcars_Disp', mtcars['disp'], mtcars['mpg'], ('Displacement', 'MPG'))
        except Exception:
            pass

        try:
            iris = rdata('datasets', 'iris')
            _try_add(datasets, 'Iris_Sepal', iris['Sepal.Length'], iris['Sepal.Width'],
                     ('SepalLength', 'SepalWidth'))
        except Exception:
            pass

        try:
            faithful = rdata('datasets', 'faithful')
            _try_add(datasets, 'Faithful', faithful['eruptions'], faithful['waiting'],
                     ('EruptionDuration', 'WaitingTime'))
        except Exception:
            pass

        try:
            swiss = rdata('datasets', 'swiss')
            _try_add(datasets, 'Swiss_Education', swiss['Education'], swiss['Fertility'],
                     ('Education', 'Fertility'))
            _try_add(datasets, 'Swiss_Agriculture', swiss['Agriculture'], swiss['Fertility'],
                     ('Agriculture', 'Fertility'))
        except Exception:
            pass

        try:
            trees = rdata('datasets', 'trees')
            _try_add(datasets, 'Trees_Girth', trees['Girth'], trees['Volume'],
                     ('Girth', 'Volume'))
        except Exception:
            pass

        try:
            airquality = rdata('datasets', 'airquality')
            _try_add(datasets, 'AirQuality_Temp', airquality['Temp'], airquality['Ozone'],
                     ('Temperature', 'Ozone'))
            _try_add(datasets, 'AirQuality_Wind', airquality['Wind'], airquality['Ozone'],
                     ('Wind', 'Ozone'))
        except Exception:
            pass

        # --- ISLR package ---
        try:
            auto = rdata('ISLR', 'Auto')
            _try_add(datasets, 'Auto_HP', auto['horsepower'], auto['mpg'],
                     ('Horsepower', 'MPG'))
            _try_add(datasets, 'Auto_Weight', auto['weight'], auto['mpg'],
                     ('Weight', 'MPG'))
            _try_add(datasets, 'Auto_Displacement', auto['displacement'], auto['mpg'],
                     ('Displacement', 'MPG'))
        except Exception:
            pass

        # --- MASS package ---
        try:
            boston = rdata('MASS', 'Boston')
            _try_add(datasets, 'Boston_Rooms', boston['rm'], boston['medv'],
                     ('AvgRooms', 'MedianValue'))
            _try_add(datasets, 'Boston_LSTAT', boston['lstat'], boston['medv'],
                     ('LowerStatus%', 'MedianValue'))
            _try_add(datasets, 'Boston_Crime', boston['crim'], boston['medv'],
                     ('CrimeRate', 'MedianValue'))
        except Exception:
            pass

        try:
            cats = rdata('MASS', 'cats')
            _try_add(datasets, 'Cats_BodyWt', cats['Bwt'], cats['Hwt'],
                     ('BodyWeight', 'HeartWeight'))
        except Exception:
            pass

        # --- carData package ---
        try:
            prestige = rdata('carData', 'Prestige')
            _try_add(datasets, 'Prestige_Income', prestige['income'], prestige['prestige'],
                     ('Income', 'Prestige'))
            _try_add(datasets, 'Prestige_Education', prestige['education'], prestige['prestige'],
                     ('Education', 'Prestige'))
        except Exception:
            pass

        # --- ggplot2 package ---
        try:
            diamonds = rdata('ggplot2', 'diamonds')
            idx = np.random.default_rng(42).choice(len(diamonds), 2000, replace=False)
            d_sub = diamonds.iloc[idx]
            _try_add(datasets, 'Diamonds_Carat', d_sub['carat'], d_sub['price'],
                     ('Carat', 'Price'))
            _try_add(datasets, 'Diamonds_Depth', d_sub['depth'], d_sub['price'],
                     ('Depth', 'Price'))
        except Exception:
            pass

        try:
            midwest = rdata('ggplot2', 'midwest')
            _try_add(datasets, 'Midwest_Poverty', midwest['percbelowpoverty'], midwest['percollege'],
                     ('PovertyRate', 'CollegeRate'))
            _try_add(datasets, 'Midwest_PopDens', midwest['popdensity'], midwest['percollege'],
                     ('PopDensity', 'CollegeRate'))
        except Exception:
            pass

        try:
            econ = rdata('ggplot2', 'economics')
            _try_add(datasets, 'Economics_Unemp', econ['unemploy'], econ['pce'],
                     ('Unemployment', 'PersonalConsumption'))
        except Exception:
            pass

        try:
            mpg = rdata('ggplot2', 'mpg')
            _try_add(datasets, 'MPG_Displ', mpg['displ'], mpg['hwy'],
                     ('EngineDisplacement', 'HighwayMPG'))
            _try_add(datasets, 'MPG_Cty', mpg['cty'], mpg['hwy'],
                     ('CityMPG', 'HighwayMPG'))
        except Exception:
            pass

        # --- wooldridge package (Econometrics) ---
        try:
            wage1 = rdata('wooldridge', 'wage1')
            _try_add(datasets, 'Wage1_Educ', wage1['educ'], wage1['wage'],
                     ('Education', 'HourlyWage'))
            _try_add(datasets, 'Wage1_Exper', wage1['exper'], wage1['wage'],
                     ('Experience', 'HourlyWage'))
        except Exception:
            pass

        try:
            wage2 = rdata('wooldridge', 'wage2')
            _try_add(datasets, 'Wage2_IQ', wage2['IQ'], wage2['wage'],
                     ('IQ', 'MonthlyWage'))
            _try_add(datasets, 'Wage2_Educ', wage2['educ'], wage2['wage'],
                     ('Education', 'MonthlyWage'))
        except Exception:
            pass

        try:
            hprice1 = rdata('wooldridge', 'hprice1')
            _try_add(datasets, 'HPrice_Sqrft', hprice1['sqrft'], hprice1['price'],
                     ('SqFt', 'HousePrice'))
            _try_add(datasets, 'HPrice_Lotsize', hprice1['lotsize'], hprice1['price'],
                     ('LotSize', 'HousePrice'))
        except Exception:
            pass

        try:
            ceosal1 = rdata('wooldridge', 'ceosal1')
            _try_add(datasets, 'CEO_Sales', ceosal1['sales'], ceosal1['salary'],
                     ('FirmSales', 'CEOSalary'))
            _try_add(datasets, 'CEO_ROE', ceosal1['roe'], ceosal1['salary'],
                     ('ROE', 'CEOSalary'))
        except Exception:
            pass

        try:
            rdchem = rdata('wooldridge', 'rdchem')
            _try_add(datasets, 'RDChem_Sales', rdchem['sales'], rdchem['rd'],
                     ('Sales', 'RDSpending'))
        except Exception:
            pass

        try:
            meap01 = rdata('wooldridge', 'meap01')
            _try_add(datasets, 'MEAP_Expend', meap01['exppp'], meap01['math4'],
                     ('ExpenditurePerPupil', 'Math4thGrade'))
        except Exception:
            pass

        # --- Ecdat package (Econometrics) ---
        try:
            crime = rdata('Ecdat', 'Crime')
            _try_add(datasets, 'Crime_Density', crime['density'], crime['crmrte'],
                     ('PopDensity', 'CrimeRate'))
            _try_add(datasets, 'Crime_Wage', crime['wcon'], crime['crmrte'],
                     ('ConstructionWage', 'CrimeRate'))
        except Exception:
            pass

        try:
            gasoline = rdata('Ecdat', 'Gasoline')
            _try_add(datasets, 'Gasoline_Income', gasoline['lincomep'], gasoline['lgaspcar'],
                     ('LogIncome', 'LogGasPerCar'))
        except Exception:
            pass

        # --- gapminder ---
        try:
            gm = rdata('gapminder', 'gapminder')
            gm2007 = gm[gm['year'] == 2007]
            _try_add(datasets, 'Gapminder_GDP', gm2007['gdpPercap'], gm2007['lifeExp'],
                     ('GDPperCapita', 'LifeExpectancy'))
            _try_add(datasets, 'Gapminder_Pop', gm2007['pop'], gm2007['lifeExp'],
                     ('Population', 'LifeExpectancy'))
        except Exception:
            pass

        # --- openintro ---
        try:
            bw = rdata('openintro', 'babies')
            _try_add(datasets, 'Babies_BirthWt', bw['gestation'], bw['bwt'],
                     ('GestationDays', 'BirthWeight'))
        except Exception:
            pass

        try:
            evals = rdata('openintro', 'evals')
            _try_add(datasets, 'Evals_Beauty', evals['bty_avg'], evals['score'],
                     ('BeautyRating', 'TeachingScore'))
        except Exception:
            pass

        # --- DAAG package ---
        try:
            ais = rdata('DAAG', 'ais')
            _try_add(datasets, 'AIS_BMI', ais['BMI'], ais['pcBfat'],
                     ('BMI', 'PercentBodyFat'))
        except Exception:
            pass

        # --- psych package ---
        try:
            sat = rdata('psych', 'sat.act')
            _try_add(datasets, 'SAT_ACT', sat['SATV'], sat['ACT'],
                     ('SATVerbal', 'ACTScore'))
        except Exception:
            pass

    except ImportError:
        print("  rdatasets not available — install with: pip install rdatasets")
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
    print(f"    Naive:   {res_df['naive_se_error_pct'].mean():.1f}%")
    print(f"    HC3:     {res_df['hc3_se_error_pct'].mean():.1f}%")
    print(f"    AMRI v1: {res_df['amri_se_error_pct'].mean():.1f}%")
    print(f"    AMRI v2: {res_df['amri_v2_se_error_pct'].mean():.1f}%")

    # Average bootstrap coverage
    print(f"\n  Average bootstrap coverage (target: 0.95):")
    print(f"    Naive:   {res_df['naive_boot_coverage'].mean():.4f}")
    print(f"    HC3:     {res_df['hc3_boot_coverage'].mean():.4f}")
    print(f"    AMRI v1: {res_df['amri_boot_coverage'].mean():.4f}")
    print(f"    AMRI v2: {res_df['amri_v2_boot_coverage'].mean():.4f}")

    # Who's closest to 0.95?
    res_df['naive_dist_95'] = abs(res_df['naive_boot_coverage'] - 0.95)
    res_df['hc3_dist_95'] = abs(res_df['hc3_boot_coverage'] - 0.95)
    res_df['amri_dist_95'] = abs(res_df['amri_boot_coverage'] - 0.95)
    res_df['amri_v2_dist_95'] = abs(res_df['amri_v2_boot_coverage'] - 0.95)

    all_dists = res_df[['naive_dist_95', 'hc3_dist_95', 'amri_dist_95', 'amri_v2_dist_95']]
    min_dist = all_dists.min(axis=1)
    amri_v2_closest = (res_df['amri_v2_dist_95'] <= min_dist + 0.005).sum()
    amri_closest = (res_df['amri_dist_95'] <= min_dist + 0.005).sum()
    hc3_closest = (res_df['hc3_dist_95'] <= min_dist + 0.005).sum()
    naive_closest = (res_df['naive_dist_95'] <= min_dist + 0.005).sum()

    print(f"\n  Closest to 0.95 coverage (within 0.005):")
    print(f"    AMRI v2: {amri_v2_closest}/{n_total} datasets")
    print(f"    AMRI v1: {amri_closest}/{n_total} datasets")
    print(f"    HC3:     {hc3_closest}/{n_total} datasets")
    print(f"    Naive:   {naive_closest}/{n_total} datasets")

    # Datasets where misspecification matters (ratio > 1.1 or < 0.9)
    misspec = res_df[res_df['se_ratio'] > 1.1]
    if len(misspec) > 0:
        print(f"\n  Datasets with detected misspecification (SE ratio > 1.1): {len(misspec)}")
        for _, row in misspec.iterrows():
            print(f"    {row['dataset']}: ratio={row['se_ratio']:.3f}, "
                  f"Naive coverage={row['naive_boot_coverage']:.4f}, "
                  f"AMRI coverage={row['amri_boot_coverage']:.4f}")

    # Statistical test: is AMRI v2 coverage closer to 0.95 than competitors?
    print(f"\n  Paired tests: AMRI v2 distance from 0.95 vs others")
    for other, col in [('Naive', 'naive_dist_95'), ('HC3', 'hc3_dist_95'), ('AMRI v1', 'amri_dist_95')]:
        diffs = res_df[col] - res_df['amri_v2_dist_95']
        t_stat, p_val = stats.ttest_1samp(diffs, 0)
        direction = "v2 closer" if diffs.mean() > 0 else "other closer"
        sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else "ns")
        print(f"    vs {other:7s}: diff={diffs.mean():+.5f} ({direction}), "
              f"t={t_stat:.3f}, p={p_val:.4f} {sig}")

    # Save
    figs = Path(__file__).resolve().parent.parent / "figures"
    res_df.to_csv(figs / 'real_data_results.csv', index=False)
    print(f"\n  Results saved to figures/real_data_results.csv")
    print("\nDONE.")
