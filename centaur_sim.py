"""Entry point to run the Centaur Fusion Loop (CFL) simulation.

Centaur Fusion Loop combines:
1) MAS OODA core (Analyst -> Strategist -> Taskmaster)
2) Industry baseline proportional rule engine (hourly stabilizer)
3) Zupervisor supervisory layer (same budget/event governance as OODA)
"""

from __future__ import annotations

import os
import csv
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

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
    apply_ooda_bid_schedule_hour_tick,
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
# Shorter lookback: react faster to recent good CPA / spend windows.
RULE_LOOKBACK_STEPS = 72
# Aggressive fusion preset (centaur_sim only): hourly steering, wider envelope,
# looser up-gates, softer down-steps — aims to reduce floor-bid lock-in vs MAS.
RULE_MIN_CLICKS_REQUIRED = 5
RULE_NORMAL_MAX_UP_PCT = 0.05
RULE_NORMAL_MAX_DN_PCT = 0.04
RULE_OPPORTUNITY_MAX_UP_PCT = 0.06
RULE_OPPORTUNITY_MAX_DN_PCT = 0.028
RULE_DAY_SPEND_RATIO_CUTOFF = 0.82
RULE_OPPORTUNITY_DAY_SPEND_RATIO_CUTOFF = 0.92
# Gates for bid-up paths (rolling spend vs daily budget)
RULE_UP_SPEND_RATIO_TIGHT = 0.95
RULE_UP_SPEND_RATIO_RELAXED = 0.88
# Down-step CPA / spend gates (fractions of target_cpa and daily budget roll-up)
RULE_DOWN_CPA_HARD_MULT = 1.28
RULE_DOWN_SPEND_HARD_MULT = 1.08
RULE_DOWN_CPA_SOFT_MULT = 1.02
RULE_DOWN_SPEND_SOFT_MULT = 0.98
# Competitive adaptation against MAS cumulative-F reference.
# When trailing badly, Centaur should recover reach faster.
RULE_TRAIL_GAP_THRESHOLD = -800.0
RULE_LEAD_GAP_THRESHOLD = 1800.0
RULE_TRAIL_UP_MULTIPLIER = 1.8
RULE_LEAD_UP_MULTIPLIER = 0.7
RULE_LEAD_DOWN_MULTIPLIER = 1.25
RULE_TRAIL_ZERO_LEAD_CLICK_FLOOR = 24
# Holiday pacing safeguards: avoid burning full budget too early while still
# entering holiday with a minimally competitive bid.
HOLIDAY_KICKOFF_MIN_BID = 3.50
HOLIDAY_KICKOFF_HOURS = 3
HOLIDAY_PACING_BUFFER = 0.08
HOLIDAY_PACING_MAX_MULT = 1.20
HOLIDAY_PACING_MIN_STEP_DOWN = 0.08
HOLIDAY_PACING_MAX_STEP_DOWN = 0.45

DEBUG_LOG_INTERVAL_HOURS = int(os.getenv("CENTAUR_DEBUG_LOG_INTERVAL_HOURS", "6"))
DEBUG_COMPARE_START_HOUR = int(os.getenv("CENTAUR_DEBUG_COMPARE_START_HOUR", "168"))
DEBUG_GAP_STOP_THRESHOLD = os.getenv("CENTAUR_DEBUG_GAP_STOP_THRESHOLD")
RUN_HEARTBEAT_HOURS = int(os.getenv("CENTAUR_RUN_HEARTBEAT_HOURS", "1"))
# Centaur-vs-MAS runtime debug guardrails (off by default).
DEBUG_MODE_ENABLED = os.getenv("CENTAUR_DEBUG_MODE", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
DEBUG_NEGATIVE_GAP_STREAK_HOURS = int(
    os.getenv("CENTAUR_DEBUG_NEGATIVE_GAP_STREAK_HOURS", "48")
)
DEBUG_HOURLY_GAP_STOP_THRESHOLD = float(
    os.getenv("CENTAUR_DEBUG_HOURLY_GAP_STOP_THRESHOLD", "-1275.0")
)
DEBUG_CUM_GAP_STOP_THRESHOLD = float(
    os.getenv("CENTAUR_DEBUG_CUM_GAP_STOP_THRESHOLD", "-9300.0")
)
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


def _load_mas_hourly_reference() -> Dict[int, float]:
    """Load hour -> hourly F for MAS from optimization CSV."""
    if not os.path.exists(MAS_CUMULATIVE_REFERENCE_PATH):
        return {}

    mas_hourly_by_hour: Dict[int, float] = {}
    prev_cumulative: Optional[float] = None
    try:
        with open(MAS_CUMULATIVE_REFERENCE_PATH, newline="") as f:
            reader = csv.DictReader(f)
            mas_rows = [row for row in reader if row.get("source") == "MAS"]
        mas_rows.sort(key=lambda row: int(float(row["hour"])))
        for row in mas_rows:
            hour = int(float(row["hour"]))
            cumulative = float(row["cumulative_f"])
            hourly = cumulative if prev_cumulative is None else cumulative - prev_cumulative
            mas_hourly_by_hour[hour] = hourly
            prev_cumulative = cumulative
    except (ValueError, KeyError):
        return {}
    return mas_hourly_by_hour


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
            "min_bid": max(0.50, target_bid * 0.92),
            "max_bid": min(20.0, target_bid * 1.14),
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
    # Hourly micro-steering everywhere so the rule layer can track MAS-like windows.
    check_every = 1
    return {
        "enabled": True,
        "mode": mode,
        "target_bid": target_bid,
        "min_bid": max(0.50, target_bid * 0.85),
        "max_bid": min(20.0, target_bid * 1.14),
        "allowed_direction": "up_only" if is_recovery else "both",
        "check_every_hours": check_every,
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
    cumulative_gap_vs_mas: Optional[float] = None,
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
    trailing = (
        cumulative_gap_vs_mas is not None
        and cumulative_gap_vs_mas <= RULE_TRAIL_GAP_THRESHOLD
    )
    leading = (
        cumulative_gap_vs_mas is not None
        and cumulative_gap_vs_mas >= RULE_LEAD_GAP_THRESHOLD
    )
    up_mult = RULE_TRAIL_UP_MULTIPLIER if trailing else RULE_LEAD_UP_MULTIPLIER if leading else 1.0
    down_mult = RULE_LEAD_DOWN_MULTIPLIER if leading else 1.0

    if allowed_direction in ("both", "up_only"):
        if avg_cpa <= target_cpa and rolling_spend_ratio < RULE_UP_SPEND_RATIO_TIGHT:
            candidate_bid = current_max_bid * (
                1.0 + float(policy["max_step_up_pct"]) * up_mult
            )
        elif avg_cpa <= target_cpa * 1.05 and rolling_spend_ratio < RULE_UP_SPEND_RATIO_RELAXED:
            candidate_bid = current_max_bid * (
                1.0 + float(policy["max_step_up_pct"]) * 0.5 * up_mult
            )

    if allowed_direction == "both":
        if (
            avg_cpa >= target_cpa * RULE_DOWN_CPA_HARD_MULT
            or rolling_spend_ratio > RULE_DOWN_SPEND_HARD_MULT
        ):
            candidate_bid = current_max_bid * (
                1.0 - float(policy["max_step_down_pct"]) * down_mult
            )
        elif (
            avg_cpa > target_cpa * RULE_DOWN_CPA_SOFT_MULT
            or rolling_spend_ratio > RULE_DOWN_SPEND_SOFT_MULT
        ):
            candidate_bid = current_max_bid * (
                1.0 - float(policy["max_step_down_pct"]) * 0.5 * down_mult
            )

    # If we are trailing and the recent hour produced no leads despite meaningful
    # click volume, force a stronger upward probe to avoid low-bid stagnation.
    if trailing and total_leads == 0 and total_clicks >= RULE_TRAIL_ZERO_LEAD_CLICK_FLOOR:
        candidate_bid = max(
            candidate_bid,
            current_max_bid * (1.0 + float(policy["max_step_up_pct"]) * 2.0),
        )

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


def _apply_holiday_pacing_guard(
    env: SandboxEnv,
    state_history: List[SimulationState],
    current_hour: int,
    scheduler_event: str,
    last_holiday_guard_fire_hour: Optional[int],
) -> bool:
    """
    Keep holiday spend paced through the day.

    In the latest regressions, Centaur often exhausted budget early during
    holiday days, then flat-lined while MAS still converted late-day traffic.
    """
    if scheduler_event != "HOLIDAY":
        return False
    if not state_history:
        return False
    if last_holiday_guard_fire_hour == current_hour:
        return False

    latest = state_history[-1]
    daily_budget = float(latest.biz_inputs.daily_budget)
    current_bid = float(latest.biz_inputs.max_bid)
    current_day_spend = float(latest.derived_variables.current_day_spend or 0.0)
    if daily_budget <= 0 or current_bid <= 0:
        return False
    if latest.derived_variables.budget_status != "normal":
        return False

    hour_of_day = current_hour % 24
    day_spend_ratio = current_day_spend / daily_budget

    # Kickoff floor: avoid entering holiday with a non-competitive bid.
    if (
        hour_of_day <= HOLIDAY_KICKOFF_HOURS
        and day_spend_ratio <= 0.35
        and current_bid < HOLIDAY_KICKOFF_MIN_BID
    ):
        new_bid = round(HOLIDAY_KICKOFF_MIN_BID, 2)
        env.configure(max_bid=new_bid)
        print(
            f"[Centaur Holiday Guard] h={current_hour:>3} kickoff "
            f"bid {current_bid:.2f} -> {new_bid:.2f}",
            flush=True,
        )
        return True

    # Pace budget by virtual time: if spending materially ahead of time,
    # trim bid to preserve runway for later holiday hours.
    expected_ratio = min(1.0, (hour_of_day + 1) / 24.0)
    max_reasonable_ratio = min(
        1.0, expected_ratio * HOLIDAY_PACING_MAX_MULT + HOLIDAY_PACING_BUFFER
    )
    if day_spend_ratio <= max_reasonable_ratio:
        return False

    overshoot = day_spend_ratio - max_reasonable_ratio
    step_down = min(
        HOLIDAY_PACING_MAX_STEP_DOWN,
        max(HOLIDAY_PACING_MIN_STEP_DOWN, overshoot * 0.95),
    )
    new_bid = round(max(0.50, current_bid * (1.0 - step_down)), 2)
    if new_bid >= round(current_bid, 2):
        return False

    env.configure(max_bid=new_bid)
    print(
        f"[Centaur Holiday Guard] h={current_hour:>3} hod={hour_of_day:>2} "
        f"spend_ratio={day_spend_ratio:.2f} cap={max_reasonable_ratio:.2f} "
        f"bid {current_bid:.2f} -> {new_bid:.2f}",
        flush=True,
    )
    return True


def run_centaur_fusion_simulation(
    total_steps: int = TOTAL_STEPS,
    initial_max_bid: float = INITIAL_MAX_BID,
    initial_daily_budget: float = INITIAL_DAILY_BUDGET,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[List[SimulationState], dict]:
    """Run CFL and return (effective_state_history, aggregate_metrics)."""
    _clear_centaur_log_files()

    scheduler = VolatilityScheduler()
    env = SandboxEnv(scheduler=scheduler, seed=seed)
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
    mas_hourly_by_hour = _load_mas_hourly_reference() if DEBUG_MODE_ENABLED else {}
    gap_stop_threshold = (
        float(DEBUG_GAP_STOP_THRESHOLD)
        if DEBUG_GAP_STOP_THRESHOLD is not None
        else None
    )
    cumulative_f_centaur = 0.0
    cumulative_gap_vs_mas: Optional[float] = None
    negative_cumulative_gap_streak = 0
    last_logged_hour = -1

    active_tech_ping: Optional[Dict[str, Any]] = None
    prev_tech_state: str = "NORMAL"
    ooda_rule_policy: Optional[Dict[str, Any]] = None
    last_rule_fire_hour: Optional[int] = None
    last_holiday_guard_fire_hour: Optional[int] = None
    mas_bid_schedule: Optional[List[float]] = None
    mas_schedule_next_idx: int = 0
    last_ooda_context: Optional[Dict[str, Any]] = None

    for step in range(total_steps):
        env.act()
        state_history = env.observe()
        current_hour = env.clock.current_hour
        sched_event = scheduler.get_v_multiplier(current_hour)["event"]
        if (step + 1) % (max(1, RUN_HEARTBEAT_HOURS) * STEPS_PER_HOUR) == 0:
            latest = state_history[-1]
            progress_pct = 100.0 * (step + 1) / max(1, total_steps)
            print(
                f"[Centaur Run] step={step + 1}/{total_steps} ({progress_pct:5.1f}%) "
                f"day={latest.market_outcome.current_day:>2} hour={current_hour:>3} "
                f"event={sched_event:<7} "
                f"bid={latest.biz_inputs.max_bid:>5.2f} "
                f"day_spend={latest.derived_variables.current_day_spend:>7.2f}/"
                f"{latest.biz_inputs.daily_budget:>7.2f}",
                flush=True,
            )
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

            # Debug mode: compare utility with MAS at the current virtual timestamp
            # and enforce protective early-stop conditions.
            if DEBUG_MODE_ENABLED and latest_f is not None:
                mas_hourly = mas_hourly_by_hour.get(current_hour)
                mas_cum = mas_cumulative_by_hour.get(current_hour)
                timestamp = f"day={env.clock.current_day:>2} hour={current_hour:>3}"
                if mas_hourly is not None:
                    hourly_gap = latest_f - mas_hourly
                    print(
                        f"[Centaur Debug Compare] {timestamp} "
                        f"hourlyF(CENTAUR)={latest_f:>9.2f} "
                        f"hourlyF(MAS)={mas_hourly:>9.2f} gap={hourly_gap:>9.2f} "
                        f"neg_cum_streak={negative_cumulative_gap_streak:>3}",
                        flush=True,
                    )
                    if hourly_gap <= DEBUG_HOURLY_GAP_STOP_THRESHOLD:
                        print(
                            f"[Centaur Debug Compare] Early stop: hourly gap {hourly_gap:.2f} "
                            f"<= threshold {DEBUG_HOURLY_GAP_STOP_THRESHOLD:.2f}",
                            flush=True,
                        )
                        break
                else:
                    print(
                        f"[Centaur Debug Compare] {timestamp} MAS hourly reference unavailable",
                        flush=True,
                    )

                if mas_cum is not None:
                    cumulative_gap_vs_mas = cumulative_f_centaur - mas_cum
                    if cumulative_gap_vs_mas < 0:
                        negative_cumulative_gap_streak += 1
                    else:
                        negative_cumulative_gap_streak = 0
                    print(
                        f"[Centaur Debug Compare] {timestamp} "
                        f"cumF(CENTAUR)={cumulative_f_centaur:>10.2f} "
                        f"cumF(MAS)={mas_cum:>10.2f} gap={cumulative_gap_vs_mas:>10.2f}",
                        flush=True,
                    )
                    if (
                        negative_cumulative_gap_streak
                        >= max(1, DEBUG_NEGATIVE_GAP_STREAK_HOURS)
                    ):
                        print(
                            "[Centaur Debug Compare] Early stop: cumulative gap stayed "
                            f"negative for {negative_cumulative_gap_streak} hours",
                            flush=True,
                        )
                        break
                    if cumulative_gap_vs_mas <= DEBUG_CUM_GAP_STOP_THRESHOLD:
                        print(
                            "[Centaur Debug Compare] Early stop: cumulative gap "
                            f"{cumulative_gap_vs_mas:.2f} <= threshold "
                            f"{DEBUG_CUM_GAP_STOP_THRESHOLD:.2f}",
                            flush=True,
                        )
                        break
                else:
                    print(
                        f"[Centaur Debug Compare] {timestamp} MAS cumulative reference unavailable",
                        flush=True,
                    )

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
                        cumulative_gap_vs_mas = gap
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
        # Hourly fusion rules: suppress only during active crash wait/active crash,
        # not during RECOVERY_PENDING (post-crash bid ramp before OODA tick).
        suppress_hourly_rule = curr_tech_state in ("CRASH_ACTIVE", "CRASH_PENDING")
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
        ooda_fired_this_step = False

        if (
            (regular_ooda_tick or is_crash_newly_active or is_recovery_confirmed)
            and len(state_history) >= STEPS_PER_HOUR
        ):
            ooda_fired_this_step = True
            schedule_hours = (
                1
                if (is_crash_newly_active or is_recovery_confirmed)
                else max(1, effective_interval // STEPS_PER_HOUR)
            )
            analysis_result = analyst.analyze(
                state_history,
                tech_ping=active_tech_ping,
                previous_ooda=last_ooda_context,
                ooda_schedule_horizon_hours=schedule_hours,
            )
            virtual_hour = env.clock.current_hour % 24
            analysis_result["timestamp"] = f"2000-01-01T{virtual_hour:02d}:00:00"
            decision = strategist.decide(analysis_result, state_history)
            execution = taskmaster.execute_cycle(
                state_history, analysis_result, decision, schedule_hours=schedule_hours
            )
            mas_bid_schedule = execution.get("bid_schedule") or [
                float(execution["bid_execution"]["actual"])
            ]
            mas_schedule_next_idx = 1
            env.configure(max_bid=float(execution["bid_execution"]["actual"]))
            ooda_rule_policy = _build_ooda_rule_policy(
                decision=decision,
                executed_bid=float(execution["bid_execution"]["actual"]),
                current_hour=current_hour,
                effective_interval_hours=max(1, effective_interval // STEPS_PER_HOUR),
            )
            last_ooda_context = {
                "virtual_hour": int(env.clock.current_hour),
                "virtual_day": int(env.clock.current_day),
                "executed_anchor_bid": float(execution["bid_execution"]["actual"]),
                "mode": execution.get("mode_executed"),
                "schedule_hours": int(schedule_hours),
                "bid_schedule": list(mas_bid_schedule),
            }

            if active_tech_ping is not None and active_tech_ping.get("event") == "RECOVERY":
                active_tech_ping = None

        mas_schedule_next_idx = apply_ooda_bid_schedule_hour_tick(
            env=env,
            step=step,
            ooda_fired_this_step=ooda_fired_this_step,
            bid_schedule=mas_bid_schedule,
            schedule_next_idx=mas_schedule_next_idx,
        )

        state_history = env.observe()
        if (
            (step + 1) % HOURLY_STEP_INTERVAL == 0
            and len(state_history) >= STEPS_PER_HOUR
            and not suppress_hourly_rule
        ):
            did_fire = _apply_agentic_rule_control(
                env=env,
                state_history=state_history,
                target_cpa=TARGET_CPA_CHF,
                policy=ooda_rule_policy,
                current_hour=current_hour,
                last_rule_fire_hour=last_rule_fire_hour,
                cumulative_gap_vs_mas=cumulative_gap_vs_mas,
            )
            if did_fire:
                last_rule_fire_hour = current_hour

        state_history = env.observe()
        if (step + 1) % HOURLY_STEP_INTERVAL == 0 and sched_event == "HOLIDAY":
            did_guard = _apply_holiday_pacing_guard(
                env=env,
                state_history=state_history,
                current_hour=current_hour,
                scheduler_event=sched_event,
                last_holiday_guard_fire_hour=last_holiday_guard_fire_hour,
            )
            if did_guard:
                last_holiday_guard_fire_hour = current_hour

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

