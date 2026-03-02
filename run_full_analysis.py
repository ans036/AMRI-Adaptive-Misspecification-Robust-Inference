#!/usr/bin/env python
"""
Master runner for FULL analysis pipeline.
Runs all scripts sequentially with progress tracking.
"""
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Ensure output dirs exist
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

def run_script(name, script_path, args=None, timeout_minutes=120):
    """Run a Python script with real-time output."""
    cmd = [sys.executable, "-u", str(script_path)]
    if args:
        cmd.extend(args)

    print(f"\n{'='*78}")
    print(f"  STARTING: {name}")
    print(f"  Script: {script_path}")
    print(f"  Timeout: {timeout_minutes} minutes")
    print(f"{'='*78}\n", flush=True)

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            timeout=timeout_minutes * 60,
            capture_output=False,  # Let output go to stdout
        )
        elapsed = time.time() - t0
        status = "SUCCESS" if result.returncode == 0 else f"FAILED (code {result.returncode})"
        print(f"\n  >> {name}: {status} in {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"\n  >> {name}: TIMEOUT after {elapsed/60:.1f} min", flush=True)
        return False
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  >> {name}: ERROR: {e} after {elapsed/60:.1f} min", flush=True)
        return False


def main():
    t_global = time.time()

    print("=" * 78)
    print("  AMRI FULL ANALYSIS PIPELINE")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)
    print(flush=True)

    scripts = [
        # (name, script_path, args, timeout_minutes)
        ("Formal Theory Verification", "src/formal_theory.py", [], 60),
        ("Competitor Comparison", "src/competitor_comparison.py", [], 120),
        ("Minimax Theory", "src/minimax_theory.py", [], 60),
        ("Real Data Validation", "src/real_data_validation.py", [], 60),
        ("Statistical Guarantees", "src/statistical_guarantees.py", [], 30),
        ("Publication Figures", "src/publication_figures.py", [], 15),
    ]

    # Check which scripts already have fresh results
    results_status = {}

    results = {}
    for name, script, args, timeout in scripts:
        script_path = PROJECT_ROOT / script
        if not script_path.exists():
            print(f"  SKIP: {name} — script not found: {script}")
            results[name] = None
            continue

        ok = run_script(name, script_path, args, timeout)
        results[name] = ok

    # Final summary
    total_time = time.time() - t_global
    print(f"\n\n{'='*78}")
    print("  FULL ANALYSIS PIPELINE COMPLETE")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*78}")
    for name, ok in results.items():
        if ok is None:
            status = "SKIPPED"
        elif ok:
            status = "OK"
        else:
            status = "FAILED"
        print(f"  {status:8s}  {name}")

    print(f"\n  Results in: {RESULTS_DIR}")
    print(f"  Figures in: {FIGURES_DIR}")
    print("=" * 78)

    # Return non-zero if any failed
    if any(v is False for v in results.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()
