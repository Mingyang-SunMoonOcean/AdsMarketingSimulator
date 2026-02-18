"""Standalone script to compute and plot optimization function from simulation results."""

import os

import matplotlib.pyplot as plt
import pandas as pd

from core.volatility_scheduler import WEBSITE_CRASH_HOURS, HOLIDAY_HOURS
from logic.optimization import calculate_opti_function

WINDOW_SIZE = 4  # hours (4 steps * 15 mins)


def main():
    csv_file = "data/ib_results.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run industry_baseline_sim.py first.")
        return

    df = pd.read_csv(csv_file)
    opti_results = []

    for i in range(0, len(df), WINDOW_SIZE):
        window = df.iloc[i : i + WINDOW_SIZE]
        if len(window) < WINDOW_SIZE:
            break
        current_hour = int(window["current_hour"].mean())
        is_holiday = HOLIDAY_HOURS[0] <= current_hour <= HOLIDAY_HOURS[1]
        is_crash = WEBSITE_CRASH_HOURS[0] <= current_hour <= WEBSITE_CRASH_HOURS[1]

        opti_value = calculate_opti_function(window, is_holiday)

        opti_results.append({
            "hour": window["current_hour"].mean(),
            "opti_function": opti_value,
            "volatility": window["volatility"].mean(),
            "event": "CRASH" if is_crash else "HOLIDAY" if is_holiday else "NORMAL",
        })

    opti_df = pd.DataFrame(opti_results)

    plt.figure(figsize=(12, 6))
    plt.plot(
        opti_df["hour"],
        opti_df["opti_function"],
        label="Optimization Function $F$",
        color="#2c3e50",
    )
    plt.axvspan(
        WEBSITE_CRASH_HOURS[0],
        WEBSITE_CRASH_HOURS[1],
        color="red",
        alpha=0.15,
        label="Website Crash",
    )
    plt.axvspan(
        HOLIDAY_HOURS[0],
        HOLIDAY_HOURS[1],
        color="orange",
        alpha=0.15,
        label="Competitor Spike",
    )
    plt.title("Optimization Function Analysis: Zurich Dealership Baseline", fontsize=14)
    plt.ylabel("Optimization Function Value ($F$)")
    plt.xlabel("Simulation Time (Hours)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig("data/optimization_function_graph.png")
    opti_df.to_csv("data/optimization_function_results.csv", index=False)
    print("Success: data/optimization_function_graph.png and data/optimization_function_results.csv generated.")


if __name__ == "__main__":
    main()
