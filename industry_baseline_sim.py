from __future__ import annotations

import os

import csv
from typing import Any, List, Tuple

from sandbox_env import SandboxEnv
from state_manager import SimulationState
from volatility_scheduler import VolatilityScheduler


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
LOOKBACK_STEPS = 96          # 24 hours
MIN_CLICKS_REQUIRED = 10     # avoid reacting to noise
BID_FLOOR = 0.50             # never go below this
BID_CEILING = 20.00          # protect against runaway bids
TARGET_CPA_CHF = 80.0

# Initial config for the baseline
INITIAL_MAX_BID = 5.00
INITIAL_DAILY_BUDGET = 1000.0

# Human intervener (website crash & holiday) parameters
CRASH_START_HOUR = 120
CRASH_END_HOUR = 168
HOLIDAY_START_HOUR = 300
HOLIDAY_END_HOUR = 468

HOLIDAY_DAILY_BUDGET = 1000.0
BASELINE_MAX_BID = 6.50


def apply_proportional_rule(
    env: SandboxEnv,
    state_history: List[SimulationState],
    target_cpa: float,
) -> None:
    """
    Realistic Hybrid Rule Engine:
    1. Lookback Window: Uses a rolling 24-hour average to smooth noise.
    2. Statistical Significance: Only acts if there is enough data (min_clicks).
    3. Efficiency Rule: Lowers bid if CPA is too high.
    4. Volume Rule: Aggressively increases bid if CPA is low AND spend is under-pacing.
    5. Inventory Guard: Prevents bids from dropping below a floor or exceeding a ceiling.
    """

    # 1. Extract historical data from the rolling window
    window = state_history[-LOOKBACK_STEPS:]
    if not window:
        return

    # 2. Calculate aggregate metrics over the window
    total_spend = sum(s.market_outcome.spend for s in window)
    total_leads = sum(s.market_outcome.leads for s in window)
    total_clicks = sum(s.market_outcome.clicks for s in window)

    # Current config from the latest state snapshot
    latest = state_history[-1]
    current_max_bid = latest.biz_inputs.max_bid
    daily_budget = latest.biz_inputs.daily_budget

    # 3. Statistical guardrail – do nothing if we don't have enough data
    if total_clicks < MIN_CLICKS_REQUIRED:
        return

    # --- A. ROLLING PACING MULTIPLIER ---
    # We treat the 'daily_budget' as the target spend for any 24h window.
    pacing_multiplier = 1.0
    if total_spend < (daily_budget * 0.85):
        # Under-spending over the last 24h: increase bid to find volume
        pacing_multiplier = 1.10
    elif total_spend > (daily_budget * 1.10):
        # Over-spending over the last 24h: decrease bid to conserve
        pacing_multiplier = 0.90

    # --- B. ROLLING EFFICIENCY MULTIPLIER ---
    efficiency_multiplier = 1.0
    if total_leads > 0:
        avg_cpa = total_spend / total_leads
        if avg_cpa >= target_cpa * 1.3:
            efficiency_multiplier = 0.75  # Aggressive cut
        elif avg_cpa >= target_cpa * 1.1:
            efficiency_multiplier = 0.85  # Moderate cut
        elif avg_cpa > target_cpa:
            efficiency_multiplier = 0.95  # Minor trim
        elif avg_cpa < target_cpa * 0.8:
            efficiency_multiplier = 1.10  # Scale up
    else:
        # Death Spiral Protection: Only cut if we spent > 1.5x target with 0 leads
        if total_spend > (target_cpa * 1.5):
            efficiency_multiplier = 0.70

    new_max_bid = current_max_bid * pacing_multiplier * efficiency_multiplier

    # Enforce physical boundaries
    new_max_bid = max(BID_FLOOR, min(new_max_bid, BID_CEILING))

    # Execute only if there's a meaningful change
    if round(new_max_bid, 2) != round(current_max_bid, 2):
        env.configure(max_bid=new_max_bid)


def apply_human_intervener(
    env: SandboxEnv,
    current_hour: int,
    scheduler: VolatilityScheduler,
) -> None:
    """
    Smarter Marketer Implementation:
    1. Weekly Efficiency Review (7-day cycle)
    2. Event-based Alerting (4h/2h lag)
    3. Monthly Budget Reset (30-day cycle)
    """
    state_history: List[SimulationState] = env.observe()
    if not state_history:
        return

    latest = state_history[-1]

    # --- A. DAILY ROUTINE CHECK (Every 12 Hours) ---
    history_24h = state_history[-96:]
    if history_24h:
        rolling_spend = sum(s.market_outcome.spend for s in history_24h)
        if rolling_spend < (latest.biz_inputs.daily_budget * 0.5):
            print(f"[Human Audit] Hour {current_hour}: Low utilization detected. Resetting bid to {INITIAL_MAX_BID}.")
            env.configure(max_bid=INITIAL_MAX_BID)

    # --- B. WEEKLY EFFICIENCY REVIEW (Every 168 Hours / 7 Days) ---
    if current_hour > 0 and current_hour % 168 == 0:
        # Look at the last 7 days (672 steps)
        weekly_history = state_history[-672:]
        if weekly_history:
            total_spend = sum(s.market_outcome.spend for s in weekly_history)
            total_leads = sum(s.market_outcome.leads for s in weekly_history)
            weekly_cpa = total_spend / total_leads if total_leads > 0 else float('inf')

            print(f"[Human Weekly Review] Hour {current_hour}: Analyzing past 7 days. Weekly CPA: {weekly_cpa:.2f}")

            # RECALIBRATION: If inefficient, cut budget to 'Efficiency Floor'
            if weekly_cpa > TARGET_CPA_CHF * 1.2:
                print(f"  -> CPA too high. Reducing budget by 30% to force efficiency.")
                env.configure(daily_budget=latest.biz_inputs.daily_budget * 0.7)

            # RECALIBRATION: If very efficient, scale up to capture more volume
            elif weekly_cpa < TARGET_CPA_CHF * 0.8:
                print(f"  -> High efficiency detected. Increasing budget by 20% to scale.")
                env.configure(daily_budget=latest.biz_inputs.daily_budget * 1.2)

    # --- C. EVENT-BASED ALERTS (Asymmetric / Reactive) ---
    condition = scheduler.get_v_multiplier(current_hour)
    event = condition["event"]

    # 4-hour lag for Crash Detection
    if event == "CRASH" and current_hour >= (CRASH_START_HOUR + 4):
        if latest.biz_inputs.max_bid > 0.01:
            print(f"[Human Alert] Hour {current_hour}: Received site-down alert. Pausing ads.")
            env.configure(max_bid=0.01)

    # 2-hour lag for Recovery Verification
    if event == "NORMAL" and current_hour >= (CRASH_END_HOUR + 2):
        if latest.biz_inputs.max_bid < 0.50:
            print(f"[Human Alert] Hour {current_hour}: Verified recovery. Resuming at 50% safety budget.")
            env.configure(daily_budget=INITIAL_DAILY_BUDGET * 0.5, max_bid=INITIAL_MAX_BID)

    # --- D. STRATEGIC CALENDAR (Anticipatory) ---
    # Holiday Start: 24h lead time to increase budget
    if current_hour == (HOLIDAY_START_HOUR - 24):
        print(f"[Human Strategy] Hour {current_hour}: Preparing for holiday. Setting aggressive budget.")
        env.configure(daily_budget=HOLIDAY_DAILY_BUDGET)

    # Holiday End: 12h cooldown to reset
    if current_hour == (HOLIDAY_END_HOUR + 12):
        print(f"[Human Strategy] Hour {current_hour}: Holiday over. Resetting to baseline for efficiency.")
        env.configure(daily_budget=INITIAL_DAILY_BUDGET)


def run_industry_baseline_simulation(
    total_steps: int = TOTAL_STEPS,
    hourly_step_interval: int = HOURLY_STEP_INTERVAL,
    human_step_interval: int = HUMAN_STEP_INTERVAL,
    target_cpa: float = TARGET_CPA_CHF,
    initial_max_bid: float = INITIAL_MAX_BID,
    initial_daily_budget: float = INITIAL_DAILY_BUDGET,
) -> Tuple[List[SimulationState], dict]:
    """
    Run the 37-day "Industry Baseline" simulation.

    Two control loops operate on the same SandboxEnv instance:
    - Loop A (Proportional Rule): every virtual hour (every 4 steps).
    - Loop B (Human Intervener): every `human_step_interval` steps.

    Returns a tuple of (state_history, aggregate_metrics).
    """
    scheduler = VolatilityScheduler()
    env = SandboxEnv(scheduler=scheduler)

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

    ## the state history is 37 days but only the last 30 days are effective

    write_to_csv("ib_simulation_results.csv", effective_state_history)

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