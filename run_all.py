"""
run_all.py  —  Reproducible "run everything" orchestrator.

Runs Industry Baseline (IB), MAS (OODA), and Centaur (CFL) back-to-back
for N iterations, each seeded deterministically, then computes the
optimization function and saves all artefacts with seed+iteration postfixes.

Usage
-----
    python run_all.py                          # seed=42, runs=5 (defaults)
    python run_all.py --seed 1234              # custom seed
    python run_all.py --seed 0 --runs 3        # 3 iterations, seed=0
    python run_all.py --seed 7 --runs 1 --out-dir results/

Output layout (inside --out-dir, default: data/)
-------------------------------------------------
    ib_results_seed{S}_run{I}.csv
    mas_results_seed{S}_run{I}.csv
    centaur_results_seed{S}_run{I}.csv
    optimization_function_results_seed{S}_run{I}.csv
    aggregate_metrics_seed{S}_run{I}.csv
    optimization_function_graph_seed{S}_run{I}.png
    cumulative_f_graph_seed{S}_run{I}.png
    breakdown_graphs_seed{S}_run{I}.png

The canonical "current-run" files (data/*.csv / data/*.png) are also
overwritten after each iteration so every other tool that reads from
data/ always sees the latest result.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_for_run(base_seed: int, run_idx: int) -> int:
    """
    Derive a per-simulation-component seed from the base seed and run index.

    Uses a deterministic bit-mixing so seeds differ substantially even for
    adjacent run indices, avoiding accidental correlation.
    """
    # SplitMix64-style mix
    s = (base_seed * 6364136223846793005 + run_idx * 1442695040888963407) & 0xFFFF_FFFF_FFFF_FFFF
    s ^= s >> 30
    s = (s * 0xBF58476D1CE4E5B9) & 0xFFFF_FFFF_FFFF_FFFF
    s ^= s >> 27
    s = (s * 0x94D049BB133111EB) & 0xFFFF_FFFF_FFFF_FFFF
    s ^= s >> 31
    return int(s & 0x7FFF_FFFF)          # keep it positive / 31-bit


def _stamp(base_seed: int, run_idx: int) -> str:
    return f"seed{base_seed}_run{run_idx}"


def _stamped(base: str, stamp: str) -> str:
    """Insert stamp before file extension: foo.csv  -> foo_seed42_run0.csv"""
    root, ext = os.path.splitext(base)
    return f"{root}_{stamp}{ext}"


def _copy_to_stamped(src: str, out_dir: str, basename: str, stamp: str) -> None:
    """Copy src to out_dir/<basename>_<stamp><ext> if src exists."""
    if not os.path.exists(src):
        return
    dst = os.path.join(out_dir, _stamped(basename, stamp))
    shutil.copy2(src, dst)
    print(f"    saved → {dst}")


# ---------------------------------------------------------------------------
# Per-iteration runner
# ---------------------------------------------------------------------------

def run_one(
    base_seed: int,
    run_idx: int,
    out_dir: str,
    ib_write_path: str = "data/ib_results.csv",
    mas_write_path: str = "data/mas_results.csv",
    centaur_write_path: str = "data/centaur_results.csv",
    opti_write_path: str = "data/optimization_function_results.csv",
    metrics_write_path: str = "data/aggregate_metrics.csv",
) -> None:
    stamp = _stamp(base_seed, run_idx)
    run_seed = _seed_for_run(base_seed, run_idx)
    print(
        f"\n{'='*70}\n"
        f"  RUN {run_idx}   base_seed={base_seed}   run_seed={run_seed}   stamp={stamp}\n"
        f"{'='*70}"
    )

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Industry Baseline
    # ------------------------------------------------------------------
    print(f"\n[{stamp}] ── Industry Baseline …")
    t0 = time.perf_counter()
    from industry_baseline_sim import (
        run_industry_baseline_simulation,
        write_to_csv as ib_write_to_csv,
    )
    ib_hist, ib_metrics = run_industry_baseline_simulation(seed=run_seed)
    ib_write_to_csv(ib_write_path, ib_hist)
    print(
        f"    leads={ib_metrics['total_leads']}  "
        f"spend={ib_metrics['total_spend']:.2f}  "
        f"cpa={ib_metrics['overall_cpa']}  "
        f"elapsed={time.perf_counter()-t0:.1f}s"
    )
    _copy_to_stamped(ib_write_path, out_dir, "ib_results.csv", stamp)

    # ------------------------------------------------------------------
    # 2. MAS (OODA)
    # ------------------------------------------------------------------
    print(f"\n[{stamp}] ── MAS (OODA) …")
    t0 = time.perf_counter()
    from ooda_sim import (
        run_ooda_simulation,
        write_to_csv as ooda_write_to_csv,
    )
    mas_hist, mas_metrics = run_ooda_simulation(seed=run_seed)
    ooda_write_to_csv(mas_write_path, mas_hist)
    print(
        f"    leads={mas_metrics['total_leads']}  "
        f"spend={mas_metrics['total_spend']:.2f}  "
        f"cpa={mas_metrics['overall_cpa']}  "
        f"elapsed={time.perf_counter()-t0:.1f}s"
    )
    _copy_to_stamped(mas_write_path, out_dir, "mas_results.csv", stamp)

    # ------------------------------------------------------------------
    # 3. Centaur (CFL)
    # ------------------------------------------------------------------
    print(f"\n[{stamp}] ── Centaur (CFL) …")
    t0 = time.perf_counter()
    from centaur_sim import (
        run_centaur_fusion_simulation,
        write_to_csv as centaur_write_to_csv,
    )
    cfl_hist, cfl_metrics = run_centaur_fusion_simulation(seed=run_seed)
    centaur_write_to_csv(centaur_write_path, cfl_hist)
    print(
        f"    leads={cfl_metrics['total_leads']}  "
        f"spend={cfl_metrics['total_spend']:.2f}  "
        f"cpa={cfl_metrics['overall_cpa']}  "
        f"elapsed={time.perf_counter()-t0:.1f}s"
    )
    _copy_to_stamped(centaur_write_path, out_dir, "centaur_results.csv", stamp)

    # ------------------------------------------------------------------
    # 4. Optimization function + graphs + aggregate metrics
    # ------------------------------------------------------------------
    print(f"\n[{stamp}] ── Optimization function …")
    t0 = time.perf_counter()
    from optimization_function_calculator import (
        compute_opti_results,
        compute_daily,
        plot_opti,
        plot_cumulative,
        plot_breakdown,
        build_metrics_table,
        IB_CSV, MAS_CSV, CENTAUR_CSV,
        OUT_OPTI, OUT_CUMUL, OUT_BREAK, OUT_TABLE,
    )
    import pandas as pd

    ib_opti      = compute_opti_results(IB_CSV,      "IB")
    mas_opti     = compute_opti_results(MAS_CSV,     "MAS")
    centaur_opti = compute_opti_results(CENTAUR_CSV, "CENTAUR")

    combined = pd.concat(
        [d for d in [ib_opti, mas_opti, centaur_opti] if d is not None],
        ignore_index=True,
    )
    combined.to_csv(opti_write_path, index=False)
    print(f"    {len(combined)} hourly windows  elapsed={time.perf_counter()-t0:.1f}s")
    _copy_to_stamped(opti_write_path, out_dir, "optimization_function_results.csv", stamp)

    ib_daily      = compute_daily(IB_CSV,      "IB")
    mas_daily     = compute_daily(MAS_CSV,     "MAS")
    centaur_daily = compute_daily(CENTAUR_CSV, "CENTAUR")

    ib_raw      = pd.read_csv(IB_CSV)      if os.path.exists(IB_CSV)      else None
    mas_raw     = pd.read_csv(MAS_CSV)     if os.path.exists(MAS_CSV)     else None
    centaur_raw = pd.read_csv(CENTAUR_CSV) if os.path.exists(CENTAUR_CSV) else None

    print(f"\n[{stamp}] ── Generating plots …")
    plot_opti(ib_opti, mas_opti, centaur_opti)
    plot_cumulative(ib_opti, mas_opti, centaur_opti)
    plot_breakdown(ib_daily, mas_daily, centaur_daily)
    for src, basename in [
        (OUT_OPTI,  "optimization_function_graph.png"),
        (OUT_CUMUL, "cumulative_f_graph.png"),
        (OUT_BREAK, "breakdown_graphs.png"),
    ]:
        _copy_to_stamped(src, out_dir, basename, stamp)

    print(f"\n[{stamp}] ── Aggregate metrics …")
    metrics_df = build_metrics_table(
        ib_raw, mas_raw, centaur_raw,
        ib_opti, mas_opti, centaur_opti,
    )
    metrics_df.to_csv(metrics_write_path)
    _copy_to_stamped(metrics_write_path, out_dir, "aggregate_metrics.csv", stamp)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:,.2f}".format)
    print(metrics_df.T.to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run IB + MAS + Centaur simulations N times with a fixed seed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed for all simulations.",
    )
    p.add_argument(
        "--runs", type=int, default=5,
        help="Number of independent iterations to execute.",
    )
    p.add_argument(
        "--out-dir", dest="out_dir", default="data",
        help="Directory to write stamped output files into.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = _parse_args(argv)

    print(
        f"\n{'#'*70}\n"
        f"  run_all.py  seed={args.seed}  runs={args.runs}  out_dir={args.out_dir}\n"
        f"{'#'*70}"
    )

    total_t0 = time.perf_counter()
    for i in range(args.runs):
        run_one(
            base_seed=args.seed,
            run_idx=i,
            out_dir=args.out_dir,
        )

    elapsed = time.perf_counter() - total_t0
    print(
        f"\n{'#'*70}\n"
        f"  All {args.runs} run(s) complete in {elapsed:.1f}s.\n"
        f"  Stamped artefacts written to: {os.path.abspath(args.out_dir)}\n"
        f"{'#'*70}\n"
    )


if __name__ == "__main__":
    main()
