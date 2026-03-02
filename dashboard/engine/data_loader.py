"""Load and cache pre-computed CSV results for the dashboard."""

import functools
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


@functools.lru_cache(maxsize=1)
def load_competitor_data() -> pd.DataFrame:
    """Load results/results_competitor_comparison.csv (896 rows)."""
    f = RESULTS_DIR / "results_competitor_comparison.csv"
    if f.exists():
        return pd.read_csv(f)
    return pd.DataFrame()


@functools.lru_cache(maxsize=1)
def load_full_results() -> pd.DataFrame:
    """Load results/results_full.csv (1441 rows)."""
    f = RESULTS_DIR / "results_full.csv"
    if f.exists():
        return pd.read_csv(f)
    return pd.DataFrame()


@functools.lru_cache(maxsize=1)
def load_amri_v2_data() -> pd.DataFrame:
    """Load results/results_amri_v2.csv (361 rows, includes blending weights)."""
    f = RESULTS_DIR / "results_amri_v2.csv"
    if f.exists():
        return pd.read_csv(f)
    return pd.DataFrame()


@functools.lru_cache(maxsize=1)
def load_real_data_results() -> pd.DataFrame:
    """Load figures/real_data_results.csv (61 real dataset results)."""
    f = FIGURES_DIR / "real_data_results.csv"
    if f.exists():
        return pd.read_csv(f)
    return pd.DataFrame()


def load_theorem_data() -> dict:
    """Load theorem verification CSVs."""
    dfs = {}
    for i, name in [(1, "continuity"), (2, "coverage"), (3, "efficiency")]:
        f = RESULTS_DIR / f"results_theorem{i}_{name}.csv"
        if f.exists():
            dfs[f"theorem{i}"] = pd.read_csv(f)
    return dfs


def load_minimax_data() -> dict:
    """Load minimax theory CSVs."""
    dfs = {}
    for name in ["minimax_lower_bound", "near_uniform_coverage", "oracle_efficiency"]:
        f = RESULTS_DIR / f"results_{name}.csv"
        if f.exists():
            dfs[name] = pd.read_csv(f)
    return dfs
