"""Entry point to run Industry Baseline simulation."""

from __future__ import annotations

import os

import csv
from typing import List, Optional, Tuple, Union

import numpy as np

from core.sandbox_env import SandboxEnv
from core.state_manager import SimulationState
from core.volatility_scheduler import VolatilityScheduler
from baseline.rule_engine import apply_proportional_rule
from baseline.legacy_human import apply_human_intervener


# ---------------------------------------------------------------------------
# Global configuration constants for the Industry Baseline
# ---------------------------------------------------------------------------

# Simulation timing
STEP_MINUTES = 15
HOURS_PER_DAY = 24
STEPS_PER_HOUR = int(60 / STEP_MINUTES)          # 4
TOTAL_DAYS = 37
TOTAL_STEPS = TOTAL_DAYS * HOURS_PER_DAY * STEPS_PER_HOUR  # 3,552
EFFECTIVE_STEPS = 30 * HOURS_PER_DAY * STEPS_PER_HOUR  # 2,880

# Control loop frequencies
HOURLY_STEP_INTERVAL = STEPS_PER_HOUR            # 1 virtual hour
HUMAN_STEP_INTERVAL = 48                         # 12 virtual hours (current spec)

# Proportional rule parameters
TARGET_CPA_CHF = 80.0

# Initial config for the baseline
INITIAL_MAX_BID = 5.00
INITIAL_DAILY_BUDGET = 1000.0


def run_industry_baseline_simulation(
    total_steps: int = TOTAL_STEPS,
    hourly_step_interval: int = HOURLY_STEP_INTERVAL,
    human_step_interval: int = HUMAN_STEP_INTERVAL,
    target_cpa: float = TARGET_CPA_CHF,
    initial_max_bid: float = INITIAL_MAX_BID,
    initial_daily_budget: float = INITIAL_DAILY_BUDGET,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[List[SimulationState], dict]:
    """
    Run the 37-day "Industry Baseline" simulation.

    Two control loops operate on the same SandboxEnv instance:
    - Loop A (Proportional Rule): every virtual hour (every 4 steps).
    - Loop B (Human Intervener): every `human_step_interval` steps.

    Returns a tuple of (state_history, aggregate_metrics).
    """
    scheduler = VolatilityScheduler()
    env = SandboxEnv(scheduler=scheduler, seed=seed)

    # Initial configuration
    env.configure(daily_budget=initial_daily_budget, max_bid=initial_max_bid)

    for step in range(total_steps):
        # Advance simulation by one 15-minute step
        env.act()

        # Observe current state history
        state_history = env.observe()

        # Loop A: Proportional Rule (Automation)
        if (step + 1) % hourly_step_interval == 0:
            apply_proportional_rule(env, state_history, target_cpa)

        # Loop B: Human Intervener (Manual)
        if (step + 1) % human_step_interval == 0:
            current_hour = env.clock.current_hour
            apply_human_intervener(env, current_hour, scheduler)

    state_history = env.observe()
    effective_state_history = state_history[-EFFECTIVE_STEPS:]
    total_spend = sum(s.market_outcome.spend for s in effective_state_history)
    total_leads = sum(s.market_outcome.leads for s in effective_state_history)
    total_clicks = sum(s.market_outcome.clicks for s in effective_state_history)
    aggregate_metrics = {
        "total_spend": round(total_spend, 4),
        "total_leads": int(total_leads),
        "total_clicks": int(total_clicks),
        "overall_cpa": round(total_spend / total_leads, 4) if total_leads > 0 else None,
    }

    return effective_state_history, aggregate_metrics


def write_to_csv(results_csv_path: str, state_history: List[SimulationState]) -> None:
    """
    Write the full state history to a CSV file.

    Columns: day, hour, minute, clicks, leads, spend, cpa, realized_cpc,
             daily_budget, max_bid, volatility, budget_status, current_day_spend.
    """
    os.makedirs(os.path.dirname(results_csv_path) or ".", exist_ok=True)
    if os.path.exists(results_csv_path):
        os.remove(results_csv_path)

    if not state_history:
        return

    fieldnames = [
        "current_day", "current_hour", "current_minute",
        "clicks", "leads", "spend", "cpa", "realized_cpc",
        "daily_budget", "max_bid", "volatility",
        "budget_status", "current_day_spend",
    ]

    with open(results_csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in state_history:
            writer.writerow({
                "current_day": s.market_outcome.current_day,
                "current_hour": s.market_outcome.current_hour,
                "current_minute": s.market_outcome.current_minute,
                "clicks": s.market_outcome.clicks,
                "leads": s.market_outcome.leads,
                "spend": s.market_outcome.spend,
                "cpa": s.market_outcome.cpa,
                "realized_cpc": s.market_outcome.realized_cpc,
                "daily_budget": s.biz_inputs.daily_budget,
                "max_bid": s.biz_inputs.max_bid,
                "volatility": s.external_events_inputs.volatility,
                "budget_status": s.derived_variables.budget_status,
                "current_day_spend": s.derived_variables.current_day_spend,
            })


if __name__ == "__main__":
    effective_state_history, metrics = run_industry_baseline_simulation()

    # Output to data/ib_results.csv per structure.md
    write_to_csv("data/ib_results.csv", effective_state_history)

    latest = effective_state_history[-1] if effective_state_history else None
    print("=== Industry Baseline Simulation (30 days) ===")
    if latest:
        print(f"Final virtual day: {latest.market_outcome.current_day}")
        print(f"Final virtual hour: {latest.market_outcome.current_hour}")
        print(f"Final max_bid: {latest.biz_inputs.max_bid}")
        print(f"Final daily_budget: {latest.biz_inputs.daily_budget}")
    print("--- Aggregates ---")
    print(f"Total spend: {metrics['total_spend']}")
    print(f"Total leads: {metrics['total_leads']}")
    print(f"Total clicks: {metrics['total_clicks']}")
    print(f"Overall CPA: {metrics['overall_cpa']}")
