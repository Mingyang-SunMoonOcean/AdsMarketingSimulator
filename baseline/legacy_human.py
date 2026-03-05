"""12-hour/24-hour intervention logic for Industry Baseline (legacy human marketer)."""

from typing import List

from core.sandbox_env import SandboxEnv
from core.volatility_scheduler import VolatilityScheduler, WEBSITE_CRASH_HOURS, HOLIDAY_HOURS

# Human intervener parameters (aligned with volatility_scheduler)
CRASH_START_HOUR = WEBSITE_CRASH_HOURS[0]
CRASH_END_HOUR = WEBSITE_CRASH_HOURS[1]
HOLIDAY_START_HOUR = HOLIDAY_HOURS[0]
HOLIDAY_END_HOUR = HOLIDAY_HOURS[1]

HOLIDAY_DAILY_BUDGET = 1000.0
INITIAL_MAX_BID = 5.00
INITIAL_DAILY_BUDGET = 1000.0
TARGET_CPA_CHF = 80.0


def apply_human_intervener(
    env: SandboxEnv,
    current_hour: int,
    scheduler: VolatilityScheduler,
    enable_12h_low_util_check: bool = True,
) -> None:
    """
    Smarter Marketer Implementation:
    1. Weekly Efficiency Review (7-day cycle)
    2. Event-based Alerting (4h/2h lag)
    3. Monthly Budget Reset (30-day cycle)
    """
    state_history: List = env.observe()
    if not state_history:
        return

    latest = state_history[-1]

    # --- A. DAILY ROUTINE CHECK (Every 12 Hours) ---
    # Optional switch for hybrid controllers that want baseline human behavior
    # but without periodic low-utilization bid reset.
    if enable_12h_low_util_check:
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
