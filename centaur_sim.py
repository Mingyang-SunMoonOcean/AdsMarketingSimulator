"""Entry point to run the Centaur Fusion Loop (CFL) simulation.

Centaur Fusion Loop combines:
1) MAS OODA core (Analyst -> Strategist -> Taskmaster)
2) Industry baseline proportional rule engine (hourly stabilizer)
3) Zupervisor supervisory layer (same budget/event governance as OODA)
"""

from __future__ import annotations

import os
import csv
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.sandbox_env import SandboxEnv
from core.state_manager import SimulationState
from core.volatility_scheduler import VolatilityScheduler, HOLIDAY_HOURS
from logic.optimization import calculate_opti_function
from ooda_sim import (
    Analyst,
    Strategist,
    Taskmaster,
    TechMonitor,
    Zupervisor,
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
    ZUPERVISOR_INTERVAL,
    HOLIDAY_ZUPERVISOR_INTERVAL,
    write_to_csv,
)

HOURLY_STEP_INTERVAL = STEPS_PER_HOUR
WINDOW_SIZE = STEPS_PER_HOUR
TARGET_CPA_CHF = 80.0
RULE_LOOKBACK_STEPS = 96
RULE_MIN_CLICKS_REQUIRED = 10
RULE_NORMAL_MAX_UP_PCT = 0.04
RULE_NORMAL_MAX_DN_PCT = 0.06
RULE_OPPORTUNITY_MAX_UP_PCT = 0.05
RULE_OPPORTUNITY_MAX_DN_PCT = 0.04
RULE_DAY_SPEND_RATIO_CUTOFF = 0.70
RULE_OPPORTUNITY_DAY_SPEND_RATIO_CUTOFF = 0.80

DEBUG_LOG_INTERVAL_HOURS = int(os.getenv("CENTAUR_DEBUG_LOG_INTERVAL_HOURS", "6"))
DEBUG_COMPARE_START_HOUR = int(os.getenv("CENTAUR_DEBUG_COMPARE_START_HOUR", "168"))
DEBUG_GAP_STOP_THRESHOLD = os.getenv("CENTAUR_DEBUG_GAP_STOP_THRESHOLD")
MAS_CUMULATIVE_REFERENCE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "optimization_function_results.csv",
)

_CENTAUR_LOG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "centaur",
    "logs",
)
ANALYST_LOG_PATH_CENTAUR = os.path.join(_CENTAUR_LOG_DIR, "analyst_log.jsonl")
STRATEGIST_LOG_PATH_CENTAUR = os.path.join(_CENTAUR_LOG_DIR, "strategist_log.jsonl")
TASKMASTER_LOG_PATH_CENTAUR = os.path.join(_CENTAUR_LOG_DIR, "taskmaster_log.jsonl")
ZUPERVISOR_LOG_PATH_CENTAUR = os.path.join(_CENTAUR_LOG_DIR, "zupervisor_log.jsonl")


def _clear_centaur_log_files() -> None:
    """Truncate Centaur-only logs so each run starts clean."""
    for path in [
        ANALYST_LOG_PATH_CENTAUR,
        STRATEGIST_LOG_PATH_CENTAUR,
        TASKMASTER_LOG_PATH_CENTAUR,
        ZUPERVISOR_LOG_PATH_CENTAUR,
    ]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()


def _load_mas_cumulative_reference() -> Dict[int, float]:
    """Load hour -> cumulative F for MAS (OODA) from optimization CSV."""
    if not os.path.exists(MAS_CUMULATIVE_REFERENCE_PATH):
        return {}

    mas_cumulative_by_hour: Dict[int, float] = {}
    try:
        with open(MAS_CUMULATIVE_REFERENCE_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("source") != "MAS":
                    continue
                hour = int(float(row["hour"]))
                mas_cumulative_by_hour[hour] = float(row["cumulative_f"])
    except (ValueError, KeyError):
        return {}
    return mas_cumulative_by_hour


def _compute_latest_hourly_f(state_history: List[SimulationState]) -> Optional[float]:
    """Compute optimization F for the most recent completed 1-hour window."""
    if len(state_history) < WINDOW_SIZE:
        return None

    window = state_history[-WINDOW_SIZE:]
    rows = [
        {
            "current_hour": s.market_outcome.current_hour,
            "current_day": s.market_outcome.current_day,
            "clicks": s.market_outcome.clicks,
            "leads": s.market_outcome.leads,
            "spend": s.market_outcome.spend,
            "daily_budget": s.biz_inputs.daily_budget,
            "volatility": s.external_events_inputs.volatility,
        }
        for s in window
    ]
    window_df = pd.DataFrame(rows)
    hour = int(window_df["current_hour"].mean())
    is_holiday = HOLIDAY_HOURS[0] <= hour <= HOLIDAY_HOURS[1]
    return float(calculate_opti_function(window_df, is_holiday))


def _build_ooda_rule_policy(
    decision: Dict[str, Any],
    executed_bid: float,
    current_hour: int,
    effective_interval_hours: int,
) -> Dict[str, Any]:
    """
    Build an OODA-governed rule policy envelope for the next few hours.

    The proportional layer is not a separate optimizer. It only makes bounded
    local corrections inside the corridor defined by the last OODA decision.
    """
    mode = str(decision.get("selected_mode", "NORMAL")).upper()
    target_bid = float(executed_bid)
    effective_interval_hours = max(1, effective_interval_hours)

    if mode == "EMERGENCY":
        return {
            "enabled": False,
            "mode": mode,
            "target_bid": target_bid,
            "anchor_hour": current_hour,
            "expires_at_hour": current_hour + effective_interval_hours,
        }

    if mode == "OPPORTUNITY":
        return {
            "enabled": True,
            "mode": mode,
            "target_bid": target_bid,
            "min_bid": max(0.50, target_bid * 0.96),
            "max_bid": min(20.0, target_bid * 1.08),
            "allowed_direction": "both",
            "check_every_hours": 1,
            "max_step_up_pct": RULE_OPPORTUNITY_MAX_UP_PCT,
            "max_step_down_pct": RULE_OPPORTUNITY_MAX_DN_PCT,
            "day_spend_ratio_cutoff": RULE_OPPORTUNITY_DAY_SPEND_RATIO_CUTOFF,
            "anchor_hour": current_hour,
            "expires_at_hour": current_hour + effective_interval_hours,
        }

    # Default NORMAL envelope. If the bid is still sub-competitive, only allow
    # upward trims so the fast layer helps recovery instead of stalling it.
    is_recovery = target_bid < 4.20
    return {
        "enabled": True,
        "mode": mode,
        "target_bid": target_bid,
        "min_bid": max(0.50, target_bid * 0.90),
        "max_bid": min(20.0, target_bid * 1.10),
        "allowed_direction": "up_only" if is_recovery else "both",
        "check_every_hours": max(1, effective_interval_hours // 2),
        "max_step_up_pct": RULE_NORMAL_MAX_UP_PCT,
        "max_step_down_pct": RULE_NORMAL_MAX_DN_PCT,
        "day_spend_ratio_cutoff": RULE_DAY_SPEND_RATIO_CUTOFF,
        "anchor_hour": current_hour,
        "expires_at_hour": current_hour + effective_interval_hours,
    }


def _apply_agentic_rule_control(
    env: SandboxEnv,
    state_history: List[SimulationState],
    target_cpa: float,
    policy: Optional[Dict[str, Any]],
    current_hour: int,
    last_rule_fire_hour: Optional[int],
) -> bool:
    """
    Agentic proportional controller governed by the last OODA policy envelope.
    """
    if not policy or not policy.get("enabled"):
        return False
    if current_hour <= int(policy["anchor_hour"]):
        return False
    if current_hour > int(policy["expires_at_hour"]):
        return False
    if last_rule_fire_hour == current_hour:
        return False
    if (current_hour - int(policy["anchor_hour"])) % int(policy["check_every_hours"]) != 0:
        return False

    window = state_history[-RULE_LOOKBACK_STEPS:]
    if not window:
        return False

    total_spend = sum(s.market_outcome.spend for s in window)
    total_leads = sum(s.market_outcome.leads for s in window)
    total_clicks = sum(s.market_outcome.clicks for s in window)
    latest = state_history[-1]
    current_max_bid = float(latest.biz_inputs.max_bid)
    daily_budget = float(latest.biz_inputs.daily_budget)
    current_day_spend = float(latest.derived_variables.current_day_spend or 0.0)

    if total_clicks < RULE_MIN_CLICKS_REQUIRED or current_max_bid <= 0 or daily_budget <= 0:
        return False
    if latest.derived_variables.budget_status != "normal":
        return False
    if (current_day_spend / daily_budget) >= float(policy["day_spend_ratio_cutoff"]):
        return False

    avg_cpa = (total_spend / total_leads) if total_leads > 0 else float("inf")
    rolling_spend_ratio = total_spend / daily_budget
    candidate_bid = current_max_bid
    allowed_direction = str(policy["allowed_direction"])

    if allowed_direction in ("both", "up_only"):
        if avg_cpa <= target_cpa and rolling_spend_ratio < 0.80:
            candidate_bid = current_max_bid * (1.0 + float(policy["max_step_up_pct"]))
        elif avg_cpa <= target_cpa * 1.05 and rolling_spend_ratio < 0.65:
            candidate_bid = current_max_bid * (1.0 + float(policy["max_step_up_pct"]) * 0.5)

    if allowed_direction == "both":
        if avg_cpa >= target_cpa * 1.20 or rolling_spend_ratio > 1.05:
            candidate_bid = current_max_bid * (1.0 - float(policy["max_step_down_pct"]))
        elif avg_cpa > target_cpa or rolling_spend_ratio > 0.95:
            candidate_bid = current_max_bid * (1.0 - float(policy["max_step_down_pct"]) * 0.5)

    candidate_bid = max(float(policy["min_bid"]), min(candidate_bid, float(policy["max_bid"])))
    candidate_bid = round(candidate_bid, 2)
    if candidate_bid == round(current_max_bid, 2):
        return False

    env.configure(max_bid=candidate_bid)
    print(
        f"[Centaur Rule] h={current_hour:>3} mode={policy['mode']} "
        f"bid {current_max_bid:.2f} -> {candidate_bid:.2f} "
        f"target={float(policy['target_bid']):.2f}",
        flush=True,
    )
    return True


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
    zupervisor = Zupervisor(
        policy_path=POLICY_PATH,
        shared_kb_path=SHARED_KNOWLEDGE_PATH,
        log_path=ZUPERVISOR_LOG_PATH_CENTAUR,
    )

    env.configure(daily_budget=initial_daily_budget, max_bid=initial_max_bid)
    mas_cumulative_by_hour = _load_mas_cumulative_reference()
    gap_stop_threshold = (
        float(DEBUG_GAP_STOP_THRESHOLD)
        if DEBUG_GAP_STOP_THRESHOLD is not None
        else None
    )
    cumulative_f_centaur = 0.0
    last_logged_hour = -1

    active_tech_ping: Optional[Dict[str, Any]] = None
    prev_tech_state: str = "NORMAL"
    ooda_rule_policy: Optional[Dict[str, Any]] = None
    last_rule_fire_hour: Optional[int] = None

    for step in range(total_steps):
        print(f"Step {step}, Virtual Hour {env.clock.current_hour}, Virtual Day {env.clock.current_day}")
        env.act()
        state_history = env.observe()
        current_hour = env.clock.current_hour
        sched_event = scheduler.get_v_multiplier(current_hour)["event"]
        new_ping = tech_monitor.check(step, sched_event)

        # Zupervisor-owned budget/event governance: same trigger semantics as OODA.
        budget_report = zupervisor.evaluate_budget_triggers(
            current_hour=current_hour,
            scheduler_event=sched_event,
            state_history=state_history,
        )
        if budget_report is not None:
            env.configure(daily_budget=budget_report["suggested_daily_budget"])
            suggested_max_bid = budget_report.get("suggested_max_bid")
            if suggested_max_bid is not None:
                env.configure(max_bid=float(suggested_max_bid))

        # Debug/Fairness telemetry: compare cumulative F from the same anchor hour.
        if (step + 1) % WINDOW_SIZE == 0 and len(state_history) >= WINDOW_SIZE:
            latest_f = _compute_latest_hourly_f(state_history)
            if latest_f is not None and current_hour >= DEBUG_COMPARE_START_HOUR:
                cumulative_f_centaur += latest_f

            if current_hour != last_logged_hour and current_hour >= DEBUG_COMPARE_START_HOUR:
                should_log = (
                    current_hour % max(1, DEBUG_LOG_INTERVAL_HOURS) == 0
                    or current_hour % 24 == 0
                    or current_hour == DEBUG_COMPARE_START_HOUR
                    or current_hour in (HOLIDAY_HOURS[0], HOLIDAY_HOURS[1])
                )
                if should_log:
                    mas_cum = mas_cumulative_by_hour.get(current_hour)
                    if mas_cum is not None:
                        gap = cumulative_f_centaur - mas_cum
                        print(
                            f"[Centaur Debug F] h={current_hour:>3} d={env.clock.current_day:>2} "
                            f"CENTAUR_cumF={cumulative_f_centaur:>10.2f} "
                            f"MAS_cumF={mas_cum:>10.2f} gap={gap:>10.2f}"
                        )
                        if gap_stop_threshold is not None and gap <= gap_stop_threshold:
                            print(
                                f"[Centaur Debug F] Early stop triggered: gap {gap:.2f} "
                                f"<= threshold {gap_stop_threshold:.2f}"
                            )
                            break
                    else:
                        print(
                            f"[Centaur Debug F] h={current_hour:>3} d={env.clock.current_day:>2} "
                            f"CENTAUR_cumF={cumulative_f_centaur:>10.2f} "
                            f"(MAS reference unavailable for this hour)"
                        )
                last_logged_hour = current_hour

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
            ooda_rule_policy = _build_ooda_rule_policy(
                decision=decision,
                executed_bid=float(execution["bid_execution"]["actual"]),
                current_hour=current_hour,
                effective_interval_hours=max(1, effective_interval // STEPS_PER_HOUR),
            )

            if active_tech_ping is not None and active_tech_ping.get("event") == "RECOVERY":
                active_tech_ping = None

        state_history = env.observe()
        if (
            (step + 1) % HOURLY_STEP_INTERVAL == 0
            and len(state_history) >= STEPS_PER_HOUR
            and not in_disruption
        ):
            did_fire = _apply_agentic_rule_control(
                env=env,
                state_history=state_history,
                target_cpa=TARGET_CPA_CHF,
                policy=ooda_rule_policy,
                current_hour=current_hour,
                last_rule_fire_hour=last_rule_fire_hour,
            )
            if did_fire:
                last_rule_fire_hour = current_hour

        regular_zuper_tick = (step + 1) % ZUPERVISOR_INTERVAL == 0
        holiday_zuper_tick = (
            sched_event == "HOLIDAY"
            and (step + 1) % HOLIDAY_ZUPERVISOR_INTERVAL == 0
        )
        if (regular_zuper_tick or holiday_zuper_tick) and len(state_history) >= ZUPERVISOR_INTERVAL:
            current_day = env.clock.current_day
            window = (
                state_history[-HOLIDAY_ZUPERVISOR_INTERVAL:]
                if holiday_zuper_tick and not regular_zuper_tick
                else state_history[-ZUPERVISOR_INTERVAL:]
            )
            guidance = zupervisor.provide_guidance(current_day, window)
            strategist.reload_policy()
            print(
                f"[Centaur Zupervisor] Day {current_day}: "
                f"{guidance.get('action')} — {guidance.get('reasoning')}"
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

