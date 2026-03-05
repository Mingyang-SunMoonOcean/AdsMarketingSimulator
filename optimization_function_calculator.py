"""
Compute and visualise optimization function metrics for IB, MAS (OODA), and Centaur simulations.

Outputs
-------
data/optimization_function_graph.png   — per-hour F comparison (IB vs MAS vs Centaur)
data/cumulative_f_graph.png            — cumulative F over time
data/breakdown_graphs.png              — daily leads / spend / CPA breakdown
data/aggregate_metrics.csv             — summary metrics table
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from core.volatility_scheduler import WEBSITE_CRASH_HOURS, HOLIDAY_HOURS
from logic.optimization import calculate_opti_function, TARGET_CPA

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_SIZE = 4   # 1 virtual hour = 4 × 15-min steps

IB_CSV  = "data/ib_results.csv"
MAS_CSV = "data/mas_results.csv"
CENTAUR_CSV = "data/centaur_results.csv"

OUT_OPTI  = "data/optimization_function_graph.png"
OUT_CUMUL = "data/cumulative_f_graph.png"
OUT_BREAK = "data/breakdown_graphs.png"
OUT_TABLE = "data/aggregate_metrics.csv"

# Colours
C_IB  = "#2c3e50"   # dark slate
C_MAS = "#2980b9"   # blue
C_CEN = "#8e44ad"   # purple
C_CR  = "#e74c3c"   # red  (crash zone)
C_HOL = "#f39c12"   # amber (holiday zone)

# Day-axis boundaries derived from hour boundaries
CRASH_DAY_START   = WEBSITE_CRASH_HOURS[0] / 24 + 1
CRASH_DAY_END     = WEBSITE_CRASH_HOURS[1] / 24 + 1
HOLIDAY_DAY_START = HOLIDAY_HOURS[0] / 24 + 1
HOLIDAY_DAY_END   = HOLIDAY_HOURS[1] / 24 + 1


# ---------------------------------------------------------------------------
# Helper: add event shading to an axis
# ---------------------------------------------------------------------------
def _shade_events_hours(ax, labeled: bool = True) -> None:
    kw = {"alpha": 0.12, "zorder": 0}
    ax.axvspan(WEBSITE_CRASH_HOURS[0], WEBSITE_CRASH_HOURS[1],
               color=C_CR,  label="Website Crash" if labeled else None, **kw)
    ax.axvspan(HOLIDAY_HOURS[0],       HOLIDAY_HOURS[1],
               color=C_HOL, label="Holiday Surge"  if labeled else None, **kw)


def _shade_events_days(ax, labeled: bool = True) -> None:
    kw = {"alpha": 0.12, "zorder": 0}
    ax.axvspan(CRASH_DAY_START,   CRASH_DAY_END,
               color=C_CR,  label="Website Crash" if labeled else None, **kw)
    ax.axvspan(HOLIDAY_DAY_START, HOLIDAY_DAY_END,
               color=C_HOL, label="Holiday Surge"  if labeled else None, **kw)


# ---------------------------------------------------------------------------
# Core: compute per-hour F windows
# ---------------------------------------------------------------------------
def compute_opti_results(csv_path: str, source_label: str) -> Optional[pd.DataFrame]:
    """
    Returns a DataFrame with columns:
        hour, day, opti_function, cumulative_f, volatility, event, source
    """
    if not os.path.exists(csv_path):
        print(f"  [SKIP] {csv_path} not found.")
        return None

    df = pd.read_csv(csv_path)
    records = []

    for i in range(0, len(df), WINDOW_SIZE):
        window = df.iloc[i: i + WINDOW_SIZE]
        if len(window) < WINDOW_SIZE:
            break

        h           = int(window["current_hour"].mean())
        is_holiday  = HOLIDAY_HOURS[0]       <= h <= HOLIDAY_HOURS[1]
        is_crash    = WEBSITE_CRASH_HOURS[0] <= h <= WEBSITE_CRASH_HOURS[1]
        opti_value  = calculate_opti_function(window, is_holiday)

        records.append({
            "hour":          window["current_hour"].mean(),
            "day":           window["current_day"].mean(),
            "opti_function": opti_value,
            "volatility":    window["volatility"].mean(),
            "event":         "CRASH" if is_crash else "HOLIDAY" if is_holiday else "NORMAL",
            "source":        source_label,
        })

    result_df = pd.DataFrame(records)
    result_df["cumulative_f"] = result_df["opti_function"].cumsum()
    print(f"  [{source_label}] {len(result_df)} hourly windows computed from {csv_path}")
    return result_df


# ---------------------------------------------------------------------------
# Core: compute daily aggregates from raw CSV
# ---------------------------------------------------------------------------
def compute_daily(csv_path: str, source_label: str) -> Optional[pd.DataFrame]:
    """Returns per-day sums of leads, spend, clicks and derived CPA."""
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    daily = (
        df.groupby("current_day")
        .agg(leads=("leads", "sum"), spend=("spend", "sum"),
             clicks=("clicks", "sum"))
        .reset_index()
    )
    daily["cpa"]    = daily.apply(
        lambda r: r["spend"] / r["leads"] if r["leads"] > 0 else np.nan, axis=1
    )
    daily["source"] = source_label
    return daily


# ---------------------------------------------------------------------------
# Figure 1: per-hour F
# ---------------------------------------------------------------------------
def plot_opti(ib_df, mas_df, centaur_df) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    _shade_events_hours(ax)

    if ib_df is not None:
        ax.plot(ib_df["hour"], ib_df["opti_function"],
                color=C_IB,  lw=1.4, label="Industry Baseline $F$")
    if mas_df is not None:
        ax.plot(mas_df["hour"], mas_df["opti_function"],
                color=C_MAS, lw=1.4, ls="--", label="MAS (OODA) $F$")
    if centaur_df is not None:
        ax.plot(centaur_df["hour"], centaur_df["opti_function"],
                color=C_CEN, lw=1.4, ls="-.", label="Centaur (CFL) $F$")

    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_title("Optimization Function $F$ per Hour — IB vs MAS vs Centaur", fontsize=13)
    ax.set_xlabel("Simulation Time (Virtual Hours)")
    ax.set_ylabel("$F$ Value (CHF)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_OPTI, dpi=150); plt.close(fig)
    print(f"  Saved → {OUT_OPTI}")


# ---------------------------------------------------------------------------
# Figure 2: cumulative F
# ---------------------------------------------------------------------------
def plot_cumulative(ib_df, mas_df, centaur_df) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    _shade_events_hours(ax)

    if ib_df is not None:
        ax.plot(ib_df["hour"], ib_df["cumulative_f"],
                color=C_IB,  lw=2.0, label="Industry Baseline  cumulative $F$")
    if mas_df is not None:
        ax.plot(mas_df["hour"], mas_df["cumulative_f"],
                color=C_MAS, lw=2.0, ls="--", label="MAS (OODA)  cumulative $F$")
    if centaur_df is not None:
        ax.plot(centaur_df["hour"], centaur_df["cumulative_f"],
                color=C_CEN, lw=2.0, ls="-.", label="Centaur (CFL) cumulative $F$")

    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_title("Cumulative Optimization Function — IB vs MAS vs Centaur", fontsize=13)
    ax.set_xlabel("Simulation Time (Virtual Hours)")
    ax.set_ylabel("Cumulative $F$ (CHF)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_CUMUL, dpi=150); plt.close(fig)
    print(f"  Saved → {OUT_CUMUL}")


# ---------------------------------------------------------------------------
# Figure 3: daily breakdown (leads / spend / CPA)
# ---------------------------------------------------------------------------
def plot_breakdown(ib_daily, mas_daily, centaur_daily) -> None:
    # Merge for easy alignment; keep all days present in either source
    frames = [d for d in [ib_daily, mas_daily, centaur_daily] if d is not None]
    if not frames:
        return

    # Build a unified day index
    all_days = sorted(
        set().union(*[set(d["current_day"].values) for d in frames])
    )
    days = np.array(all_days)
    bar_w = 0.26
    offsets = {"IB": -bar_w, "MAS": 0.0, "CENTAUR": bar_w}
    colors  = {"IB": C_IB, "MAS": C_MAS, "CENTAUR": C_CEN}

    def _get(df, col):
        if df is None:
            return np.zeros(len(days))
        merged = pd.DataFrame({"current_day": days}).merge(
            df[["current_day", col]], on="current_day", how="left"
        )
        return merged[col].fillna(0).values

    ib_leads  = _get(ib_daily,  "leads")
    mas_leads = _get(mas_daily, "leads")
    cen_leads = _get(centaur_daily, "leads")
    ib_spend  = _get(ib_daily,  "spend")
    mas_spend = _get(mas_daily, "spend")
    cen_spend = _get(centaur_daily, "spend")
    ib_cpa    = _get(ib_daily,  "cpa")   # NaN where leads=0
    mas_cpa   = _get(mas_daily, "cpa")
    cen_cpa   = _get(centaur_daily, "cpa")

    fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=True)
    fig.suptitle("Daily Performance Breakdown — IB vs MAS vs Centaur", fontsize=14, y=1.01)

    # ── Panel 1: Daily Leads ────────────────────────────────────────────────
    ax = axes[0]
    _shade_events_days(ax)
    if ib_daily  is not None:
        ax.bar(days + offsets["IB"],  ib_leads,  bar_w, color=C_IB,  alpha=0.85, label="IB")
    if mas_daily is not None:
        ax.bar(days + offsets["MAS"], mas_leads, bar_w, color=C_MAS, alpha=0.85, label="MAS")
    if centaur_daily is not None:
        ax.bar(days + offsets["CENTAUR"], cen_leads, bar_w, color=C_CEN, alpha=0.85, label="Centaur")
    ax.set_ylabel("Leads")
    ax.set_title("Total Leads per Day", fontsize=11)
    ax.legend(loc="upper left"); ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # ── Panel 2: Daily Spend ────────────────────────────────────────────────
    ax = axes[1]
    _shade_events_days(ax, labeled=False)
    if ib_daily  is not None:
        ax.bar(days + offsets["IB"],  ib_spend,  bar_w, color=C_IB,  alpha=0.85, label="IB")
    if mas_daily is not None:
        ax.bar(days + offsets["MAS"], mas_spend, bar_w, color=C_MAS, alpha=0.85, label="MAS")
    if centaur_daily is not None:
        ax.bar(days + offsets["CENTAUR"], cen_spend, bar_w, color=C_CEN, alpha=0.85, label="Centaur")
    ax.set_ylabel("Spend (CHF)")
    ax.set_title("Total Spend per Day", fontsize=11)
    ax.legend(loc="upper left"); ax.grid(axis="y", alpha=0.3)

    # ── Panel 3: Daily CPA ──────────────────────────────────────────────────
    ax = axes[2]
    _shade_events_days(ax, labeled=False)
    ax.axhline(TARGET_CPA, color="black", lw=1.6, ls="--",
               label=f"Target CPA = CHF {TARGET_CPA:.0f}", zorder=3)

    if ib_daily is not None:
        ib_cpa_series = np.where(ib_cpa == 0, np.nan, ib_cpa)
        ax.plot(days, ib_cpa_series, color=C_IB,  lw=1.6, marker="o",
                ms=4, label="IB actual CPA", zorder=4)

    if mas_daily is not None:
        mas_cpa_series = np.where(mas_cpa == 0, np.nan, mas_cpa)
        ax.plot(days, mas_cpa_series, color=C_MAS, lw=1.6, marker="s",
                ms=4, ls="--", label="MAS actual CPA", zorder=4)
    if centaur_daily is not None:
        cen_cpa_series = np.where(cen_cpa == 0, np.nan, cen_cpa)
        ax.plot(days, cen_cpa_series, color=C_CEN, lw=1.6, marker="^",
                ms=4, ls="-.", label="Centaur actual CPA", zorder=4)

    ax.set_xlabel("Virtual Day")
    ax.set_ylabel("CPA (CHF)")
    ax.set_title("Daily CPA vs Target", fontsize=11)
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    ax.set_xlim(days[0] - 0.8, days[-1] + 0.8)

    plt.tight_layout()
    plt.savefig(OUT_BREAK, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved → {OUT_BREAK}")


# ---------------------------------------------------------------------------
# Aggregate metrics table
# ---------------------------------------------------------------------------
def build_metrics_table(
    ib_raw, mas_raw, centaur_raw,         # raw CSVs as DataFrames
    ib_opti, mas_opti, centaur_opti,      # hourly F DataFrames
) -> pd.DataFrame:
    rows = []
    pairs = [("Industry Baseline (IB)", ib_raw, ib_opti),
             ("MAS (OODA)",             mas_raw, mas_opti),
             ("Centaur (CFL)",          centaur_raw, centaur_opti)]

    for label, raw, opti in pairs:
        if raw is None:
            continue

        leads  = int(raw["leads"].sum())
        spend  = raw["spend"].sum()
        clicks = int(raw["clicks"].sum())
        cpa    = spend / leads if leads > 0 else float("nan")
        budget_util = spend / (raw["daily_budget"].mean() * 30) * 100  # 30-day budget

        if opti is not None:
            total_f     = opti["opti_function"].sum()
            mean_f      = opti["opti_function"].mean()
            pct_pos     = (opti["opti_function"] > 0).mean() * 100
            cumul_final = opti["cumulative_f"].iloc[-1]
            f_normal    = opti.loc[opti["event"] == "NORMAL",  "opti_function"].mean()
            f_holiday   = opti.loc[opti["event"] == "HOLIDAY", "opti_function"].mean()
            f_crash     = opti.loc[opti["event"] == "CRASH",   "opti_function"].mean()
        else:
            total_f = mean_f = pct_pos = cumul_final = f_normal = f_holiday = f_crash = float("nan")

        rows.append({
            "Source":                label,
            "Total Leads":           leads,
            "Total Spend (CHF)":     round(spend, 2),
            "Total Clicks":          clicks,
            "Overall CPA (CHF)":     round(cpa, 2),
            "Budget Utilisation (%)":round(budget_util, 1),
            "Total F":               round(total_f, 0),
            "Mean F / window":       round(mean_f, 2),
            "% Positive F windows":  round(pct_pos, 1),
            "Final Cumulative F":    round(cumul_final, 0),
            "Mean F — Normal":       round(f_normal, 2),
            "Mean F — Holiday":      round(f_holiday, 2),
            "Mean F — Crash":        round(f_crash, 2),
        })

    return pd.DataFrame(rows).set_index("Source")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== Optimization Function Calculator ===\n")
    os.makedirs("data", exist_ok=True)

    # 1. Per-hour F windows
    ib_opti  = compute_opti_results(IB_CSV,  "IB")
    mas_opti = compute_opti_results(MAS_CSV, "MAS")
    centaur_opti = compute_opti_results(CENTAUR_CSV, "CENTAUR")

    if ib_opti is None and mas_opti is None and centaur_opti is None:
        print("No results files found. Run simulations first.")
        return

    # 2. Daily aggregates
    ib_daily  = compute_daily(IB_CSV,  "IB")
    mas_daily = compute_daily(MAS_CSV, "MAS")
    centaur_daily = compute_daily(CENTAUR_CSV, "CENTAUR")

    # 3. Raw for metrics table
    ib_raw  = pd.read_csv(IB_CSV)  if os.path.exists(IB_CSV)  else None
    mas_raw = pd.read_csv(MAS_CSV) if os.path.exists(MAS_CSV) else None
    centaur_raw = pd.read_csv(CENTAUR_CSV) if os.path.exists(CENTAUR_CSV) else None

    # 4. Save combined hourly CSV
    combined = pd.concat(
        [d for d in [ib_opti, mas_opti, centaur_opti] if d is not None], ignore_index=True
    )
    combined.to_csv("data/optimization_function_results.csv", index=False)
    print(f"  Combined CSV → data/optimization_function_results.csv  ({len(combined)} rows)\n")

    # 5. Plots
    print("Generating plots …")
    plot_opti(ib_opti, mas_opti, centaur_opti)
    plot_cumulative(ib_opti, mas_opti, centaur_opti)
    plot_breakdown(ib_daily, mas_daily, centaur_daily)

    # 6. Aggregate metrics
    print("\nBuilding aggregate metrics table …")
    metrics = build_metrics_table(
        ib_raw, mas_raw, centaur_raw,
        ib_opti, mas_opti, centaur_opti,
    )
    metrics.to_csv(OUT_TABLE)
    print(f"  Saved → {OUT_TABLE}\n")

    # Pretty-print to console
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:,.2f}".format)
    print("=" * 80)
    print("AGGREGATE METRICS")
    print("=" * 80)
    print(metrics.T.to_string())
    print("=" * 80)
    print("\nDone.")


if __name__ == "__main__":
    main()
