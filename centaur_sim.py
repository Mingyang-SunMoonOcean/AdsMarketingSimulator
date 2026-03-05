"""Entry point to run the Centaur Fusion Loop (CFL) simulation.

Centaur Fusion Loop combines:
1) MAS OODA core (Analyst -> Strategist -> Taskmaster)
2) Industry baseline proportional rule engine (hourly stabilizer)
3) Legacy human intervener (12-hour macro governance, without section A reset)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from core.sandbox_env import SandboxEnv
from core.state_manager import SimulationState
from core.volatility_scheduler import VolatilityScheduler, HOLIDAY_HOURS
from baseline.rule_engine import apply_proportional_rule
from baseline.legacy_human import apply_human_intervener
from ooda_sim import (
    Analyst,
    Strategist,
    Taskmaster,
    TechMonitor,
    SHARED_KNOWLEDGE_PATH,
    POLICY_PATH,
    INITIAL_MAX_BID,
    INITIAL_DAILY_BUDGET,
    TOTAL_STEPS,
    EFFECTIVE_STEPS,
    STEPS_PER_HOUR,
    OODA_STEP_INTERVAL,
    HOLIDAY_OODA_STEP_INTERVAL,
    POST_HOLIDAY_OODA_STEP_INTERVAL,
    POST_HOLIDAY_STABILIZATION_HOURS,
    write_to_csv,
)

HOURLY_STEP_INTERVAL = STEPS_PER_HOUR
HUMAN_STEP_INTERVAL = 48  # every 12 virtual hours (same cadence as baseline)
TARGET_CPA_CHF = 80.0

_CENTAUR_LOG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "centaur",
    "logs",
)
ANALYST_LOG_PATH_CENTAUR = os.path.join(_CENTAUR_LOG_DIR, "analyst_log.jsonl")
STRATEGIST_LOG_PATH_CENTAUR = os.path.join(_CENTAUR_LOG_DIR, "strategist_log.jsonl")
TASKMASTER_LOG_PATH_CENTAUR = os.path.join(_CENTAUR_LOG_DIR, "taskmaster_log.jsonl")


def _clear_centaur_log_files() -> None:
    """Truncate Centaur-only logs so each run starts clean."""
    for path in [
        ANALYST_LOG_PATH_CENTAUR,
        STRATEGIST_LOG_PATH_CENTAUR,
        TASKMASTER_LOG_PATH_CENTAUR,
    ]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()


def run_centaur_fusion_simulation(
    total_steps: int = TOTAL_STEPS,
    initial_max_bid: float = INITIAL_MAX_BID,
    initial_daily_budget: float = INITIAL_DAILY_BUDGET,
) -> Tuple[List[SimulationState], dict]:
    """Run CFL and return (effective_state_history, aggregate_metrics)."""
    _clear_centaur_log_files()

    scheduler = VolatilityScheduler()
    env = SandboxEnv(scheduler=scheduler)
    tech_monitor = TechMonitor()

    analyst = Analyst(
        shared_knowledge_path=SHARED_KNOWLEDGE_PATH,
        policy_path=POLICY_PATH,
        log_path=ANALYST_LOG_PATH_CENTAUR,
    )
    strategist = Strategist(
        shared_knowledge_path=SHARED_KNOWLEDGE_PATH,
        policy_path=POLICY_PATH,
        log_path=STRATEGIST_LOG_PATH_CENTAUR,
    )
    taskmaster = Taskmaster(
        policy_path=POLICY_PATH,
        shared_kb_path=SHARED_KNOWLEDGE_PATH,
        log_path=TASKMASTER_LOG_PATH_CENTAUR,
    )

    env.configure(daily_budget=initial_daily_budget, max_bid=initial_max_bid)

    active_tech_ping: Optional[Dict[str, Any]] = None
    prev_tech_state: str = "NORMAL"

    for step in range(total_steps):
        print(f"Step {step}, Virtual Hour {env.clock.current_hour}, Virtual Day {env.clock.current_day}")
        env.act()
        state_history = env.observe()
        current_hour = env.clock.current_hour
        sched_event = scheduler.get_v_multiplier(current_hour)["event"]
        new_ping = tech_monitor.check(step, sched_event)

        if new_ping is not None:
            active_tech_ping = new_ping
        elif tech_monitor._state == "NORMAL" and (
            active_tech_ping is not None and active_tech_ping.get("event") == "RECOVERY"
        ):
            active_tech_ping = None

        curr_tech_state = tech_monitor._state
        is_crash_newly_active = (
            curr_tech_state == "CRASH_ACTIVE" and prev_tech_state != "CRASH_ACTIVE"
        )
        is_recovery_confirmed = (
            new_ping is not None and new_ping.get("event") == "RECOVERY"
        )
        prev_tech_state = curr_tech_state

        in_disruption = curr_tech_state in ("CRASH_ACTIVE", "RECOVERY_PENDING")
        holiday_end_hour = HOLIDAY_HOURS[1]
        in_post_holiday_stabilization = (
            sched_event == "NORMAL"
            and holiday_end_hour < current_hour <= holiday_end_hour + POST_HOLIDAY_STABILIZATION_HOURS
        )
        effective_interval = (
            HOLIDAY_OODA_STEP_INTERVAL if sched_event == "HOLIDAY"
            else POST_HOLIDAY_OODA_STEP_INTERVAL if in_post_holiday_stabilization
            else OODA_STEP_INTERVAL
        )
        regular_ooda_tick = (
            not in_disruption and (step + 1) % effective_interval == 0
        )

        if (
            (regular_ooda_tick or is_crash_newly_active or is_recovery_confirmed)
            and len(state_history) >= STEPS_PER_HOUR
        ):
            analysis_result = analyst.analyze(state_history, tech_ping=active_tech_ping)
            virtual_hour = env.clock.current_hour % 24
            analysis_result["timestamp"] = f"2000-01-01T{virtual_hour:02d}:00:00"
            decision = strategist.decide(analysis_result, state_history)
            execution = taskmaster.execute_cycle(state_history, analysis_result, decision)
            env.configure(max_bid=execution["bid_execution"]["actual"])

            if active_tech_ping is not None and active_tech_ping.get("event") == "RECOVERY":
                active_tech_ping = None

        state_history = env.observe()
        if (step + 1) % HOURLY_STEP_INTERVAL == 0 and len(state_history) >= STEPS_PER_HOUR:
            apply_proportional_rule(env, state_history, TARGET_CPA_CHF)

        if (step + 1) % HUMAN_STEP_INTERVAL == 0:
            apply_human_intervener(
                env,
                current_hour,
                scheduler,
                enable_12h_low_util_check=False,
            )

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


if __name__ == "__main__":
    effective_state_history, metrics = run_centaur_fusion_simulation()
    write_to_csv("data/centaur_results.csv", effective_state_history)
    latest = effective_state_history[-1] if effective_state_history else None
    print("=== Centaur Fusion Loop Simulation (30 days) ===")
    if latest:
        print(f"Final virtual day: {latest.market_outcome.current_day}")
        print(f"Final max_bid: {latest.biz_inputs.max_bid}")
        print(f"Final daily_budget: {latest.biz_inputs.daily_budget}")
    print("--- Aggregates ---")
    print(f"Total spend: {metrics['total_spend']}")
    print(f"Total leads: {metrics['total_leads']}")
    print(f"Total clicks: {metrics['total_clicks']}")
    print(f"Overall CPA: {metrics['overall_cpa']}")

