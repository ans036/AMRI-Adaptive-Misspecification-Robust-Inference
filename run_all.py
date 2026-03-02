"""
Master script: Reproduce all AMRI results.
============================================
Usage:
  python run_all.py --quick    # Quick validation (~5 min)
  python run_all.py --full     # Full reproduction (~2-4 hours)

Outputs:
  results/          CSV files with all simulation and theory results
  figures/          Publication-quality figures and real data CSV
"""
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
RESULTS = PROJECT_ROOT / "results"
FIGURES = PROJECT_ROOT / "figures"

RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def run_step(name, cmd, timeout=600):
    """Run a single step, print status, return success."""
    print(f"\n{'='*70}")
    print(f"  STEP: {name}")
    print(f"{'='*70}")
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
            cwd=str(PROJECT_ROOT)
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            # Print last 20 lines of output
            lines = result.stdout.strip().split('\n')
            for line in lines[-20:]:
                print(f"  {line}")
            print(f"\n  DONE ({elapsed:.1f}s)")
            return True
        else:
            print(f"  FAILED (exit code {result.returncode}, {elapsed:.1f}s)")
            print(f"  stderr: {result.stderr[-500:]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        return False


def main():
    mode = '--quick' if '--quick' in sys.argv else '--full' if '--full' in sys.argv else '--quick'
    flag = '--quick' if mode == '--quick' else ''

    print(f"AMRI Reproduction Pipeline ({mode.strip('-').upper()} mode)")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {sys.executable}")
    print()

    steps = [
        ("1. Formal Theory Verification",
         f"{sys.executable} {SRC / 'formal_theory.py'} {flag}",
         300 if flag else 1800),

        ("2. Competitor Comparison",
         f"{sys.executable} {SRC / 'competitor_comparison.py'} {flag}",
         300 if flag else 3600),

        ("3. Real Data Validation (61 datasets)",
         f"{sys.executable} {SRC / 'real_data_validation.py'}",
         600),

        ("4. Statistical Guarantees",
         f"{sys.executable} {SRC / 'statistical_guarantees.py'}",
         120),
    ]

    results = {}
    t_start = time.time()

    for name, cmd, timeout in steps:
        success = run_step(name, cmd, timeout)
        results[name] = success

    total = time.time() - t_start

    # Summary
    print(f"\n{'='*70}")
    print(f"  REPRODUCTION SUMMARY ({total:.0f}s total)")
    print(f"{'='*70}")
    for name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {name}")

    n_pass = sum(results.values())
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} steps completed successfully.")

    if n_pass == n_total:
        print("\n  All results reproduced. Check results/ and figures/ directories.")
    else:
        print("\n  Some steps failed. Check output above for details.")

    return 0 if n_pass == n_total else 1


if __name__ == '__main__':
    sys.exit(main())
