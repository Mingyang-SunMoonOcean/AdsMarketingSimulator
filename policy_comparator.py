"""
Cross-run policy comparator for IB, MAS (OODA), and Centaur (CFL).

Ingests all stamped optimization-function CSVs produced by run_all.py:
    data/optimization_function_results_seed{S}_run{I}.csv

Within each run, all three policies share the same run_seed (paired design).
Outputs statistical tables and shaded superiority plots under data/.

Usage
-----
    python policy_comparator.py
    python policy_comparator.py --data-dir data --bootstrap 10000
"""

from __future__ import annotations

import argparse
import matplotlib

matplotlib.use("Agg")
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from core.volatility_scheduler import HOLIDAY_HOURS, WEBSITE_CRASH_HOURS
from optimization_function_calculator import (
    C_CEN,
    C_CR,
    C_HOL,
    C_IB,
    C_MAS,
    POST_HOLIDAY_AFTER_HOURS,
    _shade_events_hours,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SOURCES = ("IB", "MAS", "CENTAUR")
SOURCE_LABELS = {
    "IB": "Industry Baseline",
    "MAS": "MAS (OODA)",
    "CENTAUR": "Centaur (CFL)",
}
PAIRWISE = (
    ("CENTAUR", "MAS", "Centaur vs MAS"),
    ("CENTAUR", "IB", "Centaur vs IB"),
    ("MAS", "IB", "MAS vs IB"),
)
OPTi_GLOB = "optimization_function_results_seed*_run*.csv"

OUT_RUN_LEVEL = "data/policy_comparison_run_level.csv"
OUT_SUMMARY = "data/policy_comparison_summary.csv"
OUT_SEGMENT = "data/policy_comparison_segment.csv"
OUT_GAP_CUMUL = "data/policy_comparison_cumulative_gaps.png"
OUT_GAP_HOURLY = "data/policy_comparison_hourly_gaps.png"
OUT_DELTAS = "data/policy_comparison_paired_deltas.png"
OUT_FINAL_BAR = "data/policy_comparison_final_f.png"


@dataclass(frozen=True)
class RunKey:
    base_seed: int
    run_idx: int

    @property
    def stamp(self) -> str:
        return f"seed{self.base_seed}_run{self.run_idx}"


@dataclass
class PairedResult:
    better: str
    worse: str
    label: str
    n: int
    mean_diff: float
    sd_diff: float
    median_diff: float
    ci_low: float
    ci_high: float
    cohens_d: float
    wins: int
    t_stat: float
    t_p_two_sided: float
    wilcoxon_w: float
    wilcoxon_p_one_sided: float
    perm_p_one_sided: float


# ---------------------------------------------------------------------------
# Discovery & loading
# ---------------------------------------------------------------------------
def discover_opti_files(data_dir: str) -> List[str]:
    pattern = os.path.join(data_dir, OPTi_GLOB)
    files = sorted(glob.glob(pattern))
    if not files:
        fallback = os.path.join(data_dir, "optimization_function_results.csv")
        if os.path.exists(fallback):
            return [fallback]
    return files


def _parse_stamp(path: str) -> Optional[RunKey]:
    m = re.search(r"seed(\d+)_run(\d+)", path)
    if m:
        return RunKey(int(m.group(1)), int(m.group(2)))
    return None


def load_hourly_wide(path: str) -> pd.DataFrame:
    """Return wide hourly table: hour × {IB, MAS, CENTAUR} opti + cumulative."""
    df = pd.read_csv(path)
    opti = df.pivot_table(index="hour", columns="source", values="opti_function", aggfunc="first")
    cumul = df.pivot_table(index="hour", columns="source", values="cumulative_f", aggfunc="first")
    opti = opti.reindex(columns=list(SOURCES))
    cumul = cumul.reindex(columns=list(SOURCES))
    opti.columns = [f"{c}_opti" for c in opti.columns]
    cumul.columns = [f"{c}_cumul" for c in cumul.columns]
    wide = pd.concat([opti, cumul], axis=1).sort_index()
    wide.index.name = "hour"
    return wide


def build_run_level_table(files: List[str]) -> pd.DataFrame:
    rows: List[dict] = []
    for path in files:
        key = _parse_stamp(path)
        stamp = key.stamp if key else os.path.basename(path)
        base_seed = key.base_seed if key else -1
        run_idx = key.run_idx if key else -1

        df = pd.read_csv(path)
        wide = load_hourly_wide(path)

        for src in SOURCES:
            sub = df[df["source"] == src]
            if sub.empty:
                continue
            rows.append({
                "stamp": stamp,
                "base_seed": base_seed,
                "run_idx": run_idx,
                "source": src,
                "final_cumulative_f": float(sub["cumulative_f"].iloc[-1]),
                "total_f": float(sub["opti_function"].sum()),
                "mean_hourly_f": float(sub["opti_function"].mean()),
                "pct_positive_f": float((sub["opti_function"] > 0).mean() * 100),
            })

    return pd.DataFrame(rows)


def _segment_means(path: str) -> List[dict]:
    df = pd.read_csv(path)
    key = _parse_stamp(path)
    stamp = key.stamp if key else os.path.basename(path)
    post_start = float(HOLIDAY_HOURS[1] + 1)
    post_end = float(HOLIDAY_HOURS[1] + POST_HOLIDAY_AFTER_HOURS)

    rows = []
    for src in SOURCES:
        sub = df[df["source"] == src].copy()
        if sub.empty:
            continue
        seg = {
            "NORMAL": sub.loc[sub["event"] == "NORMAL", "opti_function"].mean(),
            "HOLIDAY": sub.loc[sub["event"] == "HOLIDAY", "opti_function"].mean(),
            "CRASH": sub.loc[sub["event"] == "CRASH", "opti_function"].mean(),
        }
        mask = (sub["hour"] >= post_start) & (sub["hour"] <= post_end)
        seg["POST_HOLIDAY"] = sub.loc[mask, "opti_function"].mean()
        for segment, val in seg.items():
            rows.append({
                "stamp": stamp,
                "source": src,
                "segment": segment,
                "mean_hourly_f": float(val) if pd.notna(val) else np.nan,
            })
    return rows


# ---------------------------------------------------------------------------
# Statistics (numpy-only)
# ---------------------------------------------------------------------------
def _wilcoxon_signed_rank(diffs: np.ndarray) -> Tuple[float, float]:
    """Wilcoxon W+ and one-sided p (H1: median diff > 0). Normal approx for n >= 6."""
    d = np.asarray(diffs, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 0.0, 1.0
    ranks = pd.Series(np.abs(d)).rank(method="average").values
    w_plus = float(ranks[d > 0].sum())
    mu = n * (n + 1) / 4.0
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return w_plus, 1.0
    z = (w_plus - mu) / sigma
    # one-sided P(W+ >= observed) under H0
    from math import erf, sqrt

    p_one = 0.5 * (1 - erf(z / sqrt(2)))
    return w_plus, float(np.clip(p_one, 0, 1))


def _paired_t_pvalue(diffs: np.ndarray) -> Tuple[float, float]:
    n = len(diffs)
    if n < 2:
        return 0.0, 1.0
    mean_d = float(np.mean(diffs))
    sd_d = float(np.std(diffs, ddof=1))
    if sd_d == 0:
        return (np.inf if mean_d > 0 else (-np.inf if mean_d < 0 else 0.0)), (
            0.0 if mean_d != 0 else 1.0
        )
    t = mean_d / (sd_d / np.sqrt(n))
    # Two-sided p via Student-t survival approx (adequate for thesis n ≈ 10–30)
    from math import lgamma

    df = n - 1
    x = df / (df + t * t)
    # regularized incomplete beta for t CDF
    a, b = df / 2.0, 0.5
    # use series for small df — scipy-free rough approx via normal for df>=8
    if df >= 8:
        from math import erf, sqrt

        p_two = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    else:
        # crude: compare |t| to critical values
        crit = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45,
                7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 15: 2.13, 20: 2.09, 30: 2.04}
        tc = crit.get(df, 2.0)
        p_two = 0.05 if abs(t) >= tc else 0.20
    return float(t), float(np.clip(p_two, 0, 1))


def bootstrap_ci(
    diffs: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan")
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        means[i] = sample.mean()
    alpha = (1 - ci) / 2
    return float(np.percentile(means, 100 * alpha)), float(np.percentile(means, 100 * (1 - alpha)))


def permutation_pvalue_greater(diffs: np.ndarray, n_perm: int = 10000, seed: int = 42) -> float:
    """One-sided: P(mean >= observed | H0 symmetric about 0)."""
    rng = np.random.default_rng(seed)
    obs = float(np.mean(diffs))
    n = len(diffs)
    if n == 0:
        return 1.0
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_means = (signs * diffs).mean(axis=1)
    return float((perm_means >= obs).mean())


def paired_comparison(
    pivot: pd.DataFrame,
    better: str,
    worse: str,
    label: str,
    n_boot: int = 10000,
) -> PairedResult:
    diffs = (pivot[better] - pivot[worse]).dropna().values
    n = len(diffs)
    mean_d = float(np.mean(diffs)) if n else float("nan")
    sd_d = float(np.std(diffs, ddof=1)) if n > 1 else float("nan")
    med_d = float(np.median(diffs)) if n else float("nan")
    ci_lo, ci_hi = bootstrap_ci(diffs, n_boot=n_boot) if n else (float("nan"), float("nan"))
    cohens_d = mean_d / sd_d if n > 1 and sd_d > 0 else float("nan")
    wins = int((diffs > 0).sum()) if n else 0
    t_stat, t_p_two = _paired_t_pvalue(diffs)
    w_stat, w_p_one = _wilcoxon_signed_rank(diffs)
    perm_p = permutation_pvalue_greater(diffs, n_perm=n_boot) if n else float("nan")

    return PairedResult(
        better=better,
        worse=worse,
        label=label,
        n=n,
        mean_diff=mean_d,
        sd_diff=sd_d,
        median_diff=med_d,
        ci_low=ci_lo,
        ci_high=ci_hi,
        cohens_d=cohens_d,
        wins=wins,
        t_stat=t_stat,
        t_p_two_sided=t_p_two,
        wilcoxon_w=w_stat,
        wilcoxon_p_one_sided=w_p_one,
        perm_p_one_sided=perm_p,
    )


def results_to_summary_df(results: List[PairedResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Comparison": r.label,
            "Better": SOURCE_LABELS.get(r.better, r.better),
            "Worse": SOURCE_LABELS.get(r.worse, r.worse),
            "n_runs": r.n,
            "wins": r.wins,
            "win_rate_pct": round(100 * r.wins / r.n, 1) if r.n else np.nan,
            "mean_diff_CHF": round(r.mean_diff, 2),
            "sd_diff_CHF": round(r.sd_diff, 2),
            "median_diff_CHF": round(r.median_diff, 2),
            "ci95_low_CHF": round(r.ci_low, 2),
            "ci95_high_CHF": round(r.ci_high, 2),
            "cohens_d_paired": round(r.cohens_d, 3),
            "t_statistic": round(r.t_stat, 3),
            "t_p_two_sided": round(r.t_p_two_sided, 4),
            "wilcoxon_p_one_sided": round(r.wilcoxon_p_one_sided, 4),
            "perm_p_one_sided": round(r.perm_p_one_sided, 4),
            "significant_95_perm": r.perm_p_one_sided < 0.05 if r.n else False,
        })
    return pd.DataFrame(rows)


def segment_paired_table(files: List[str], n_boot: int) -> pd.DataFrame:
    seg_rows: List[dict] = []
    for path in files:
        seg_rows.extend(_segment_means(path))
    seg_df = pd.DataFrame(seg_rows)
    if seg_df.empty:
        return seg_df

    summary_rows = []
    segments = sorted(seg_df["segment"].unique())
    for better, worse, label in PAIRWISE:
        for segment in segments:
            sub = seg_df[seg_df["segment"] == segment]
            pivot = sub.pivot_table(index="stamp", columns="source", values="mean_hourly_f")
            if better not in pivot.columns or worse not in pivot.columns:
                continue
            diffs = (pivot[better] - pivot[worse]).dropna().values
            if len(diffs) == 0:
                continue
            ci_lo, ci_hi = bootstrap_ci(diffs, n_boot=n_boot)
            summary_rows.append({
                "Comparison": label,
                "Segment": segment,
                "n_runs": len(diffs),
                "mean_diff_CHF": round(float(np.mean(diffs)), 2),
                "ci95_low_CHF": round(ci_lo, 2),
                "ci95_high_CHF": round(ci_hi, 2),
                "wins": int((diffs > 0).sum()),
            })
    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Time-series aggregation across runs
# ---------------------------------------------------------------------------
def aggregate_gap_trajectories(
    files: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (hourly_gaps, cumulative_gaps) DataFrames indexed by hour with
    columns for each pairwise gap: mean, ci_low, ci_high, frac_positive.
    """
    hourly_parts: List[pd.DataFrame] = []
    cumul_parts: List[pd.DataFrame] = []

    for path in files:
        key = _parse_stamp(path)
        stamp = key.stamp if key else path
        wide = load_hourly_wide(path)
        if not all(f"{s}_opti" in wide.columns for s in SOURCES):
            continue
        h = wide[[f"{s}_opti" for s in SOURCES]].copy()
        h.columns = list(SOURCES)
        c = wide[[f"{s}_cumul" for s in SOURCES]].copy()
        c.columns = list(SOURCES)

        gaps_h = pd.DataFrame({
            "MAS-IB": h["MAS"] - h["IB"],
            "CENTAUR-MAS": h["CENTAUR"] - h["MAS"],
            "CENTAUR-IB": h["CENTAUR"] - h["IB"],
        }, index=h.index)
        gaps_h["stamp"] = stamp
        hourly_parts.append(gaps_h)

        gaps_c = pd.DataFrame({
            "MAS-IB": c["MAS"] - c["IB"],
            "CENTAUR-MAS": c["CENTAUR"] - c["MAS"],
            "CENTAUR-IB": c["CENTAUR"] - c["IB"],
        }, index=c.index)
        gaps_c["stamp"] = stamp
        cumul_parts.append(gaps_c)

    if not hourly_parts:
        return pd.DataFrame(), pd.DataFrame()

    hourly_all = pd.concat(hourly_parts)
    cumul_all = pd.concat(cumul_parts)

    def _agg(gap_name: str, frame: pd.DataFrame) -> pd.DataFrame:
        sub = frame.reset_index().groupby("hour")[gap_name]
        return pd.DataFrame({
            "mean": sub.mean(),
            "ci_low": sub.quantile(0.025),
            "ci_high": sub.quantile(0.975),
            "frac_positive": frame.reset_index().groupby("hour")[gap_name].apply(
                lambda x: (x > 0).mean()
            ),
        })

    hourly_stats = {g: _agg(g, hourly_all) for g in ["MAS-IB", "CENTAUR-MAS", "CENTAUR-IB"]}
    cumul_stats = {g: _agg(g, cumul_all) for g in ["MAS-IB", "CENTAUR-MAS", "CENTAUR-IB"]}

    return hourly_stats, cumul_stats


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
# Distinct from Centaur purple — teal for full-stack Centaur vs baseline gap
C_CEN_IB = "#16a085"

_GAP_STYLE = {
    "CENTAUR-IB": {"color": C_CEN_IB, "label": "Centaur − IB"},
    "CENTAUR-MAS": {"color": C_CEN, "label": "Centaur − MAS"},
    "MAS-IB": {"color": C_MAS, "label": "MAS − IB"},
}


def plot_gap_panels(
    gap_stats: Dict[str, pd.DataFrame],
    title: str,
    y_label: str,
    out_path: str,
    *,
    reference_line: bool = True,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(title, fontsize=14, y=1.01)

    for ax, gap_key in zip(axes, ["MAS-IB", "CENTAUR-MAS", "CENTAUR-IB"]):
        stats = gap_stats.get(gap_key)
        if stats is None or stats.empty:
            ax.set_visible(False)
            continue
        _shade_events_hours(ax, labeled=(ax is axes[0]))
        style = _GAP_STYLE[gap_key]
        hours = stats.index.values
        mean = stats["mean"].values
        lo = stats["ci_low"].values
        hi = stats["ci_high"].values
        frac = stats["frac_positive"].values

        ax.fill_between(hours, lo, hi, color=style["color"], alpha=0.25,
                        label="95% CI across runs")
        ax.plot(hours, mean, color=style["color"], lw=2.0, label="Mean gap")
        if reference_line:
            ax.axhline(0, color="grey", lw=0.9, ls=":", zorder=1)

        ax2 = ax.twinx()
        ax2.plot(hours, frac * 100, color="#27ae60", lw=1.2, ls="--", alpha=0.85,
                 label="% runs with gap > 0")
        ax2.set_ylabel("% runs winning", fontsize=9)
        ax2.set_ylim(0, 105)
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

        ax.set_ylabel(y_label)
        ax.set_title(style["label"], fontsize=11)
        ax.grid(True, alpha=0.3)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Simulation Time (Virtual Hours)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_paired_deltas(pivot: pd.DataFrame, results: List[PairedResult], out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Paired Final Cumulative $F$ Gaps per Run", fontsize=13)

    gap_cols = [
        ("MAS", "IB", C_MAS),
        ("CENTAUR", "MAS", C_CEN),
        ("CENTAUR", "IB", C_CEN_IB),
    ]
    for ax, (better, worse, color), res in zip(axes, gap_cols, results):
        diffs = (pivot[better] - pivot[worse]).dropna()
        x = np.arange(len(diffs))
        colors = [color if d > 0 else "#95a5a6" for d in diffs]
        ax.bar(x, diffs.values, color=colors, alpha=0.85, edgecolor="white", lw=0.5)
        ax.axhline(0, color="grey", lw=0.9)
        ax.axhline(res.mean_diff, color="black", lw=1.2, ls="--",
                   label=f"mean = {res.mean_diff:,.0f}")
        ax.set_title(f"{SOURCE_LABELS[better]}\n− {SOURCE_LABELS[worse]}", fontsize=10)
        ax.set_xlabel("Run index")
        ax.set_ylabel("Δ cumulative $F$ (CHF)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.text(
            0.02, 0.98,
            f"wins {res.wins}/{res.n}\n95% CI [{res.ci_low:,.0f}, {res.ci_high:,.0f}]\n"
            f"perm p = {res.perm_p_one_sided:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_final_f_bars(pivot: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    means = [pivot[s].mean() for s in SOURCES]
    cis = []
    for s in SOURCES:
        vals = pivot[s].dropna().values
        cis.append(bootstrap_ci(vals, n_boot=5000))
    colors = [C_IB, C_MAS, C_CEN]
    labels = [SOURCE_LABELS[s] for s in SOURCES]
    x = np.arange(3)
    bars = ax.bar(x, means, color=colors, alpha=0.85, edgecolor="white", width=0.55)
    for i, (lo, hi) in enumerate(cis):
        ax.errorbar(x[i], means[i], yerr=[[means[i] - lo], [hi - means[i]]],
                    fmt="none", color="black", capsize=6, lw=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Final cumulative $F$ (CHF)")
    ax.set_title("Final Cumulative $F$ — Mean ± 95% Bootstrap CI Across Runs")
    ax.grid(axis="y", alpha=0.3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{m:,.0f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(data_dir: str = "data", n_boot: int = 10000) -> None:
    os.makedirs(data_dir, exist_ok=True)
    files = discover_opti_files(data_dir)
    if not files:
        print(f"No optimization CSVs found in {data_dir!r}. Run run_all.py first.")
        return

    print(f"=== Policy Comparator ({len(files)} runs) ===\n")
    for f in files:
        print(f"  • {f}")

    run_level = build_run_level_table(files)
    policy_rows = run_level[~run_level["source"].str.startswith("_")].copy()
    pivot = policy_rows.pivot_table(
        index=["stamp", "base_seed", "run_idx"],
        columns="source",
        values="final_cumulative_f",
    )
    pivot = pivot.reindex(columns=list(SOURCES))

    results = [
        paired_comparison(pivot, better, worse, label, n_boot=n_boot)
        for better, worse, label in PAIRWISE
    ]
    summary = results_to_summary_df(results)
    segment = segment_paired_table(files, n_boot=n_boot)

    out_run = os.path.join(data_dir, os.path.basename(OUT_RUN_LEVEL))
    out_sum = os.path.join(data_dir, os.path.basename(OUT_SUMMARY))
    out_seg = os.path.join(data_dir, os.path.basename(OUT_SEGMENT))
    policy_rows.to_csv(out_run, index=False)
    summary.to_csv(out_sum, index=False)
    if not segment.empty:
        segment.to_csv(out_seg, index=False)

    print(f"\n  Saved → {out_run}")
    print(f"  Saved → {out_sum}")
    if not segment.empty:
        print(f"  Saved → {out_seg}")

    hourly_stats, cumul_stats = aggregate_gap_trajectories(files)
    if cumul_stats:
        print("\nGenerating plots …")
        plot_gap_panels(
            cumul_stats,
            "Cumulative $F$ Outperformance (Mean Gap ± 95% CI Across Runs)",
            "Cumulative gap (CHF)",
            os.path.join(data_dir, os.path.basename(OUT_GAP_CUMUL)),
        )
        plot_gap_panels(
            hourly_stats,
            "Hourly $F$ Outperformance (Mean Gap ± 95% CI Across Runs)",
            "Hourly gap (CHF)",
            os.path.join(data_dir, os.path.basename(OUT_GAP_HOURLY)),
        )
        plot_paired_deltas(pivot, results, os.path.join(data_dir, os.path.basename(OUT_DELTAS)))
        plot_final_f_bars(pivot, os.path.join(data_dir, os.path.basename(OUT_FINAL_BAR)))

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print("\n" + "=" * 80)
    print("PAIRWISE SUPERIORITY (paired on shared run_seed)")
    print("=" * 80)
    print(summary.to_string(index=False))
    print("=" * 80)
    if not segment.empty:
        print("\nSEGMENT MEAN HOURLY F GAPS (paired)")
        print(segment.to_string(index=False))
        print("=" * 80)
    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare IB / MAS / Centaur across all runs.")
    parser.add_argument("--data-dir", default="data", help="Directory with stamped CSVs")
    parser.add_argument(
        "--bootstrap", type=int, default=10000,
        help="Bootstrap / permutation iterations",
    )
    args = parser.parse_args()
    run(data_dir=args.data_dir, n_boot=args.bootstrap)


if __name__ == "__main__":
    main()
