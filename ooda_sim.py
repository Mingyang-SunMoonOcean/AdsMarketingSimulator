"""Entry point to run Phase 2 OODA MAS simulation."""

from __future__ import annotations

import os

import csv
from typing import List, Tuple

from core.sandbox_env import SandboxEnv
from core.state_manager import SimulationState
from core.volatility_scheduler import VolatilityScheduler
from agents.analyst import Analyst
from agents.strategist import Strategist
from agents.executor import Executor
from agents.human_supervisor import HumanSupervisor


# ---------------------------------------------------------------------------
# Configuration (mirror Industry Baseline for comparability)
# ---------------------------------------------------------------------------
STEP_MINUTES = 15
HOURS_PER_DAY = 24
STEPS_PER_HOUR = int(60 / STEP_MINUTES)
TOTAL_DAYS = 37
TOTAL_STEPS = TOTAL_DAYS * HOURS_PER_DAY * STEPS_PER_HOUR
EFFECTIVE_STEPS = 30 * HOURS_PER_DAY * STEPS_PER_HOUR

INITIAL_MAX_BID = 5.00
INITIAL_DAILY_BUDGET = 1000.0


def run_ooda_simulation(
    total_steps: int = TOTAL_STEPS,
    initial_max_bid: float = INITIAL_MAX_BID,
    initial_daily_budget: float = INITIAL_DAILY_BUDGET,
) -> Tuple[List[SimulationState], dict]:
    """
    Run the OODA MAS simulation (Phase 2).

    Uses Analyst (observation), Strategist (policy + shadow pricing),
    Executor (sandbox interface), and Human Supervisor (3-7 day guidance).

    Returns a tuple of (state_history, aggregate_metrics).
    """
    scheduler = VolatilityScheduler()
    env = SandboxEnv(scheduler=scheduler)

    analyst = Analyst()
    strategist = Strategist()
    executor = Executor(env)
    human_supervisor = HumanSupervisor()

    env.configure(daily_budget=initial_daily_budget, max_bid=initial_max_bid)

    for step in range(total_steps):
        env.act()
        state_history = env.observe()

        # OODA loop (placeholder: agents are stubs; Executor already has config)
        observation = analyst.observe(state_history)
        current_hour = env.clock.current_hour
        current_day = (current_hour // 24) + 1

        # Human Supervisor: 3-7 day cadence (e.g., every 5 days)
        guidance = None
        if current_day > 0 and current_day % 5 == 0:
            guidance = human_supervisor.get_guidance(current_day, observation)

        # Strategist: policy selection
        policy = strategist.select_policy(observation, guidance)
        if policy:
            if "max_bid" in policy:
                executor.configure(max_bid=policy["max_bid"])
            if "daily_budget" in policy:
                executor.configure(daily_budget=policy["daily_budget"])

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
    """Write state history to CSV."""
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
    effective_state_history, metrics = run_ooda_simulation()

    write_to_csv("data/mas_results.csv", effective_state_history)

    latest = effective_state_history[-1] if effective_state_history else None
    print("=== OODA MAS Simulation (Phase 2, 30 days) ===")
    if latest:
        print(f"Final virtual day: {latest.market_outcome.current_day}")
        print(f"Final max_bid: {latest.biz_inputs.max_bid}")
        print(f"Final daily_budget: {latest.biz_inputs.daily_budget}")
    print("--- Aggregates ---")
    print(f"Total spend: {metrics['total_spend']}")
    print(f"Total leads: {metrics['total_leads']}")
    print(f"Total clicks: {metrics['total_clicks']}")
    print(f"Overall CPA: {metrics['overall_cpa']}")
